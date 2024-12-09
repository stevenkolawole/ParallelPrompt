#include "openai.hpp"

#include <chrono>
#include <future>
#include <string>
#include <optional>
#include <regex>
#include <iostream>
#include <fstream>
#include <numeric>
#include <unistd.h> // For getopt
#include <getopt.h>

using namespace std;
using namespace std::chrono;

string replace_in_string(const string & original, const string & toReplace, const string & replacement) {
    regex re(toReplace);
    return regex_replace(original, re, replacement);
}

string escape_json(const string& input) {
    string result;
    for (char ch : input) {
        switch (ch) {
          case '"': result += "\\\""; break;
          case '\\': result += "\\\\"; break;
          case '\b': result += "\\b"; break;
          case '\f': result += "\\f"; break;
          case '\n': result += "\\n"; break;
          case '\r': result += "\\r"; break;
          case '\t': result += "\\t"; break;
          default: result += ch; break;
        }
    }
    return result;
}

vector<nlohmann::json> error_handling(string filepath) {
    ifstream ifs(filepath);
    nlohmann::json prompts_list;
    if (!ifs.is_open()) { // making sure that filepath is valid
      cerr << "Error opening the file." << endl;
      return prompts_list;
    }

    try {
      ifs >> prompts_list;
    } catch (const nlohmann::json::parse_error& e) {
      cerr << "JSON parse error: " << e.what() << endl;
    }

  return prompts_list;
}

string getIthLetter(int i) {
  // Ensure the input is within the valid range
  int idx = i % 26;
    
  // 'A' is the first letter in ASCII and its value is 65.
  // Subtract 1 from i to get 0-based index and add to 'A'
  char letter = 'A' + idx;
  return string(1, letter); // Convert char to string
}

openai::Json call_openai(string system_prompt, string prompt, int max_tokens) {
  auto openai_instance = openai::OpenAI();
    string request = R"({
       "model": "gpt-4-0125-preview",
       "messages": [{"role": "system", "content": ")" + system_prompt + R"("}, {"role": "user", "content": ")" + prompt + R"("}],
       "max_tokens": )" + to_string(max_tokens) + R"(,
       "temperature": 0.7
    })";
    auto json_request = nlohmann::json::parse(request);

    for (int retry = 0; retry < 5; ++retry) {
    try {
        auto completion = openai_instance.chat.create(json_request);
        return completion;
    } catch (const std::exception& e) {
        if (retry < 5 - 1) {
            int delay = (1 << retry) * 1000; // Exponential backoff
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        } else {
            std::cerr << "Max retries reached. Failing." << e.what() << endl;
            throw;
        }
      }
    }
    throw std::runtime_error("Unexpected error in call_openai");
}

pair<int, string> serial(nlohmann::json formatted_prompts, string task) {
    string serial_prompt = formatted_prompts["serial"];
    string escaped_prompt = escape_json(serial_prompt);
    string system_prompt;
    if (task == "keyword_extraction"){
      system_prompt = "You are a helpful assistant specializing in keyword extraction. Do not include any irrelevant information";
    } 
    else if (task == "reading_comprehension"){
      system_prompt = "You are a helpful assistant specializing in reading comprehension. Provide concise and accurate answers based on the given context";
    }
    else {
      system_prompt = "You are a helpful assistant.";
    }
    auto completion = call_openai(system_prompt, escaped_prompt, 4000);
    string output = completion["choices"][0]["message"]["content"];
    return make_pair(completion["usage"]["completion_tokens"], output);
}

tuple<vector<int>, int, vector<pair<int, openai::Json>>> parallel(nlohmann::json formatted_prompts, string task) {
    int n = 0;
    string prompt_template = formatted_prompts["template"];
    string template_escaped = escape_json(prompt_template);
    string template_with_context; 
    if (formatted_prompts.contains("context") && !formatted_prompts["context"].empty()) {
    	string context = formatted_prompts["context"];
    	string escaped_context = escape_json(context);
    	string context_regex = "\\{context\\}"; // Escape the curly braces for the regex
    	template_with_context = replace_in_string(template_escaped, context_regex, escaped_context);
    } else {
        template_with_context = template_escaped;
    }
    string system_prompt;
    if (task == "keyword_extraction"){
      system_prompt = "You are a helpful assistant specializing in keyword extraction. Only extract values for the given keyword and do not include any irrelevant information";
      n = formatted_prompts["data"].size();
    } else if (task == "reading_comprehension"){
      system_prompt = "You are a helpful assistant specializing in reading comprehension. Provide extremely concise and accurate answers based on the given context";
      n = formatted_prompts["data"].size();
    } else if (task == "generate_n") {
      n = formatted_prompts["n"];
    }
    else {
      system_prompt = "You are a helpful assistant. Provide accurate and relevant information based on the given task";
    }
    
    vector<future<pair<long long, openai::Json>>> futures;
    for (int i = 0; i < n; ++i) {
        string prompt = template_with_context;
        if (task == "generate_n") {
          system_prompt = "You are a helpful assistant.  Provide concise and accurate answers based on the given context and do not include irrelevant information. Try to make your response start with the letter " + getIthLetter(i);
          string n_regex = "\\{n\\}"; // Escape the curly braces for the regex
          prompt = replace_in_string(prompt, n_regex, to_string(1));
        }
        else if (formatted_prompts.contains("data")) {
          string data_regex = "\\{data\\}"; // Escape the curly braces for the regex
          string escaped_data = escape_json(formatted_prompts["data"][i]);
          prompt = replace_in_string(prompt, data_regex, escaped_data);
        }
        futures.push_back(async(launch::async, [system_prompt, prompt]() {
            auto start = high_resolution_clock::now();
            auto completion = call_openai(system_prompt, prompt, 1000);
            auto end = high_resolution_clock::now();
            milliseconds duration = duration_cast<milliseconds>(end - start);
            return make_pair(duration.count(), completion);
        }));
    }

    int sum_parallel_tokens = 0;
    vector<int> tokens_list;
    vector<pair<int, openai::Json>> outputs;
    for (auto & f : futures) {
        auto [duration, completion] = f.get();
        outputs.push_back(make_pair(duration, completion["choices"][0]["message"]["content"]));
        int tokens = completion["usage"]["completion_tokens"].get<int>();
        tokens_list.push_back(tokens);
        sum_parallel_tokens += tokens;
    }
    return make_tuple(tokens_list, sum_parallel_tokens, outputs);
}

// Function to print usage information
void printUsage() {
    std::cout << "Usage: program -f <type> <value>" << std::endl;
    std::cout << "  -f <type>    Specify the type ('keyword' or 'rc')" << std::endl;
    std::cout << "  <value>      Value corresponding to the type" << std::endl;
    std::cout << "  If type is 'keyword', <value> should be a string." << std::endl;
    std::cout << "  If type is 'rc', <value> should be an integer." << std::endl;
}

int main(int argc, char* argv[]) {
    std::string queries;
    std::string task;
    std::string output;

    // Define the options
    struct option long_options[] = {
        {"queries", required_argument, nullptr, 'q'},
        {"task", required_argument, nullptr, 't'},
        {"output", required_argument, nullptr, 'o'},
        {nullptr, 0, nullptr, 0},
    };

    // Valid task options
    const std::string task_options[] = {"reading_comprehension", "keyword_extraction", "generate_n"};

    int opt;
    int option_index = 0;

    // Parse the options
    while ((opt = getopt_long(argc, argv, "q:t:o:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'q':
                queries = optarg;
                break;
            case 't':
                task = optarg;
                // Validate task option
                if (std::find(std::begin(task_options), std::end(task_options), task) == std::end(task_options)) {
                    std::cerr << "Invalid task option. Valid options are: reading_comprehension, keyword_extraction, generate_n." << std::endl;
                    return 1;
                }
                break;
            case 'o':
                output = optarg;
                break;
            case '?':
                std::cerr << "Unknown option." << std::endl;
                return 1;
            default:
                std::cerr << "Usage: " << argv[0] << " --queries <query_string> --task <task_option> --output <output file>" << std::endl;
                return 1;
        }
    }

    // Check that all required options are provided
    if (queries.empty()) {
        std::cerr << "--queries is required." << std::endl;
        return 1;
    }

    if (task.empty()) {
        std::cerr << "--task is required." << std::endl;
        return 1;
    }

    if (output.empty()) {
        std::cerr << "--output is required." << std::endl;
    }

    // Display the parsed arguments
    std::cout << "Queries: " << queries << std::endl;
    std::cout << "Task: " << task << std::endl;
    std::cout << "Output location : " << output << std::endl;

  nlohmann::json prompts_list = error_handling(queries);
  nlohmann::json results_json = nlohmann::json::array();

  double total_serial_duration = 0, total_parallel_duration = 0;
  int total_serial_tokens = 0, total_parallel_tokens = 0;
  int task_count = 0;

  for (auto& prompt : prompts_list) {
    // if (task_count >= 25) break; // Only process the first 10

    nlohmann::json result_entry;

    result_entry["prompt"] = prompt["original"];

    cout << prompt["original"] << endl;

    // Serial execution
    auto start_serial = high_resolution_clock::now();
    auto [serial_tokens, serial_output] = serial(prompt, task);
    auto end_serial = high_resolution_clock::now();
    milliseconds serial_duration = duration_cast<milliseconds>(end_serial - start_serial);

    // Parallel execution
    auto start_parallel = high_resolution_clock::now();
    auto [parallel_tokens, sum_parallel_tokens, parallel_results] = parallel(prompt, task);
    auto end_parallel = high_resolution_clock::now();
    milliseconds parallel_duration = duration_cast<milliseconds>(end_parallel - start_parallel);

    // Record results
    total_serial_duration += serial_duration.count();
    total_parallel_duration += parallel_duration.count();
    total_serial_tokens += serial_tokens;
    total_parallel_tokens += sum_parallel_tokens;

    // Log results for this prompt
    result_entry["serial_output"] = serial_output;
    result_entry["serial_num_tokens"] = serial_tokens;

    vector<string> parallel_outputs;
    vector<int> parallel_durations;
    for (const auto& [duration, output] : parallel_results) {
        parallel_outputs.push_back(output);
        parallel_durations.push_back(duration);
    }

    result_entry["parallel_output"] = parallel_outputs;
    result_entry["parallel_num_tokens"] = parallel_tokens;
    result_entry["total_parallel_tokens"] = sum_parallel_tokens;
    result_entry["serial_duration_ms"] = serial_duration.count();
    result_entry["parallel_duration_ms"] = parallel_durations;
    result_entry["total_parallel_duration_ms"] = parallel_duration.count();

    // Calculate speedup
    result_entry["speedup"] = static_cast<double>(serial_duration.count()) / parallel_duration.count();

    // Calculate normalized speedup for a fair comparison across tokens
    result_entry["normalized_speedup"] = (static_cast<double>(serial_duration.count()) / serial_tokens) / 
    (static_cast<double>(parallel_duration.count()) / sum_parallel_tokens);
    
    // Add result entry to results array
    results_json.push_back(result_entry);
    ++task_count;
  }

  // Calculate and print averages
  nlohmann::json averages;
  averages["avg_serial_duration"] = total_serial_duration / task_count;
  averages["avg_parallel_duration"] = total_parallel_duration / task_count;
  averages["avg_serial_tokens"] = total_serial_tokens / task_count;
  averages["avg_parallel_tokens"] = total_parallel_tokens / task_count;

  // Calculate average speedup
  auto speedup = static_cast<double>(total_serial_duration) / total_parallel_duration;
  double normalized_speedup = (static_cast<double>(total_serial_duration) / total_serial_tokens) /
                              (static_cast<double>(total_parallel_duration) / total_parallel_tokens);

  averages["speedup"] = speedup;
  averages["normalized_speedup"] = normalized_speedup;

  results_json.push_back({"averages", averages});

  ofstream json_file(output);
  json_file << results_json.dump(2);
  json_file.close();

  cout << "Results saved to " << output << endl;
  cout << "Average Speedup: " << speedup << "x" << endl;
  cout << "Average Normalized speedup: " << normalized_speedup << "x" << endl;

  return 0;
}