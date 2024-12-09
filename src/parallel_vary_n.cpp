#include "openai.hpp"

#include <chrono>
#include <future>
#include <string>
#include <optional>
#include <regex>
#include <iostream>
#include <fstream>
#include <numeric>

using namespace std;
using namespace std::chrono;

ofstream log_file("out/serial_vs_n_variations.txt");

string replace_in_string(const string & original, const string & toReplace, const string & replacement) {
    regex re(toReplace);
    return regex_replace(original, re, replacement);
}

string getIthLetter(int i) {
  // Ensure the input is within the valid range
  int idx = i % 26;
    
  // 'A' is the first letter in ASCII and its value is 65.
  // Subtract 1 from i to get 0-based index and add to 'A'
  char letter = 'A' + idx;
  return string(1, letter); // Convert char to string
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


openai::Json call_openai(string system_prompt, string prompt, int max_tokens) {
  auto openai_instance = openai::OpenAI();
    // cout << "Assembling request..." << endl;
    string request = R"({
       "model": "gpt-4-1106-preview",
       "messages": [{"role": "system", "content": ")" + system_prompt + R"("}, {"role": "user", "content": ")" + prompt + R"("}],
       "max_tokens": )" + to_string(max_tokens) + R"(,
       "temperature": 0.7
    })";
    // cout << request << endl;
    auto json_request = nlohmann::json::parse(request);

    // cout << "Calling openai..." << endl;
    for (int retry = 0; retry < 5; ++retry) {
    try {
        auto completion = openai_instance.chat.create(json_request);
        return completion;
    } catch (const std::exception& e) {
        std::cerr << "API call failed (attempt " << retry + 1 << "): " << e.what();
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


int serial(nlohmann::json formatted_prompts, int n) {
    string template_str = formatted_prompts["template"];
    string escaped_template = escape_json(template_str);

    string n_regex = "\\{n\\}"; // Escape the curly braces for the regex
    string replaced_template = replace_in_string(escaped_template, n_regex, to_string(n));

    if ( not formatted_prompts["context"].is_null() ) {
      string context_regex = "\\{context\\}"; // Escape the curly braces for the regex
      string escaped_context = escape_json(formatted_prompts["context"]);
      replaced_template = replace_in_string(replaced_template, context_regex, escaped_context);
    }
      
    // cout << "Calling... " << replaced_template << endl;

    string system_prompt = "You are a helpful assistant.";
    auto completion = call_openai(system_prompt, replaced_template, 4000);

    log_file << completion["choices"][0]["message"]["content"] << endl;
    return completion["usage"]["completion_tokens"];
}

int parallel(nlohmann::json formatted_prompts, int n) {
    string template_str = formatted_prompts["template"];
    string escaped_template = escape_json(template_str);

    string n_regex = "\\{n\\}"; // Escape the curly braces for the regex
    string replaced_template = replace_in_string(escaped_template, n_regex, to_string(1));

    if ( not formatted_prompts["context"].is_null() ) {
      string context_regex = "\\{context\\}"; // Escape the curly braces for the regex
      string escaped_context = escape_json(formatted_prompts["context"]);
      replaced_template = replace_in_string(replaced_template, context_regex, escaped_context);
    }

    vector<future<openai::Json>> futures;
    for (int i = 0; i < n; ++i) {
      string system_prompt = "You are a helpful assistant. Try to make your response start with the letter " + getIthLetter(i);
      futures.push_back(async(launch::async,
                              call_openai,
                              system_prompt,
                              replaced_template,
                              1000));
    }

    int parallel_tokens = 0;
    vector<openai::Json> results;
    for (auto & f : futures) {
      auto completion = f.get();
      results.push_back(completion);
      log_file << completion["choices"][0]["message"]["content"] << endl;
      parallel_tokens += completion["usage"]["completion_tokens"].get<int>();
    }

   return parallel_tokens;
}

int main() {
  nlohmann::json prompts_list = error_handling("prompts/generate_n_subset.json");
  ofstream log_file("out/serial_vs_n_variations.txt");

  int task_count;

  vector<vector<int>> all_parallel_durations;
  vector<vector<int>> all_parallel_tokens_counts;
  vector<vector<int>> all_serial_durations;
  vector<vector<int>> all_serial_tokens_counts;
  
  cout << "iterating prompts" << endl;
  for (int i = 0; i <= 50; i++) {
    // parallel execution
    vector<int> parallel_durations;
    vector<int> parallel_tokens_counts;
    vector<int> serial_durations;
    vector<int> serial_tokens_counts;
    task_count = 0;

    for (auto & prompt : prompts_list) {
      cout << "iterating" << endl;
      if (task_count >= 5) break; // Only process the first 5

      log_file << "Prompt " << (task_count + 1) << ":" << endl;
      // log_file << prompt.dump(4) << endl;
      // cout << "Prompt " << (task_count + 1) << ":" << endl;
      // cout << prompt.dump(4) << endl;

      cout << "Running " << i << endl;

      // Serial execution
      auto start_serial = high_resolution_clock::now();
      int serial_tokens = serial(prompt, i);
      auto end_serial = high_resolution_clock::now();
      chrono::milliseconds serial_duration = chrono::duration_cast<chrono::milliseconds>(end_serial - start_serial);

      cout << "Done serial" << endl;

      // Log results for this prompt
      log_file << "Serial duration: " << serial_duration.count() << " ms" << endl;
      log_file << "Serial tokens: " << serial_tokens << endl;
      serial_durations.push_back(serial_duration.count());
      serial_tokens_counts.push_back(serial_tokens);

      cout << "Running parallel" << endl;
      
      auto start_parallel = high_resolution_clock::now();
      int parallel_tokens = parallel(prompt, i); // Call with variable 'n'
      auto end_parallel = high_resolution_clock::now();
      chrono::milliseconds parallel_duration = chrono::duration_cast<chrono::milliseconds>(end_parallel - start_parallel);

      cout << "Done parallel" << endl;
      
      parallel_durations.push_back(parallel_duration.count());
      parallel_tokens_counts.push_back(parallel_tokens);
    }

    all_parallel_durations.push_back(parallel_durations);
    all_parallel_tokens_counts.push_back(parallel_tokens_counts);
    all_serial_durations.push_back(serial_durations);
    all_serial_tokens_counts.push_back(serial_tokens_counts);

    ++task_count;
  }

  // Calculate and print averages for all 'n' values
  for (size_t i = 0; i < all_serial_durations.size(); i++) {
    auto & serial_durations = all_serial_durations[i];
    auto & parallel_durations = all_parallel_durations[i];
    auto & serial_tokens_counts = all_serial_tokens_counts[i];
    auto & parallel_tokens_counts = all_parallel_tokens_counts[i];

    double avg_serial_duration = std::reduce(serial_durations.begin(), serial_durations.end()) / task_count;
    double avg_parallel_duration = std::reduce(parallel_durations.begin(), parallel_durations.end()) / task_count;
    double avg_serial_tokens = std::reduce(serial_tokens_counts.begin(), serial_tokens_counts.end()) / task_count;
    double avg_parallel_tokens = std::reduce(parallel_tokens_counts.begin(), parallel_tokens_counts.end()) / task_count;

    // Save averages to log file
    log_file << "Average Serial duration: " << avg_serial_duration << " ms" << endl;
    log_file << "Average Serial tokens: " << avg_serial_tokens << endl;
  
    cout << endl << "Average Serial duration: " << avg_serial_duration << " ms" << endl;
    cout << "Average Serial tokens: " << avg_serial_tokens << endl;

    cout << "Average Parallel duration with " << (i) << " parallel calls: " << avg_parallel_duration << " ms" << endl;
    cout << "Average Parallel tokens with " << (i) << " parallel calls: " << avg_parallel_tokens << endl;

    log_file << "Average Parallel duration with " << (i) << " parallel calls: " << avg_parallel_duration << " ms" << endl;
    log_file << "Average Parallel tokens with " << (i) << " parallel calls: " << avg_parallel_tokens << endl;

    // Calculate average speedup
    auto speedup = static_cast<double>(avg_serial_duration) / avg_parallel_duration;

    // Calculate normalized speedup for a fair comparison across tokens
    double normalized_speedup = (static_cast<double>(avg_serial_duration) / avg_serial_tokens) /
                                (static_cast<double>(avg_parallel_duration) / avg_parallel_tokens);

    cout << "Average Speedup for " << (i) << " parallel calls: " << speedup << "x" << endl;
    cout << "Average Normalized speedup for " << (i) << " parallel calls: " << normalized_speedup << "x" << endl;

    log_file << "Average Speedup for " << (i) << " parallel calls: " << speedup << "x" << endl;
    log_file << "Average Normalized speedup for " << (i) << " parallel calls: " << normalized_speedup << "x" << endl;

  }

  log_file.close();
  return 0;
}
