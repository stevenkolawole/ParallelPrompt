#include "openai.hpp"
#include "nlohmann/json.hpp"

#include <chrono>
#include <future>
#include <string>
#include <optional>
#include <regex>
#include <iostream>
#include <fstream>
#include <numeric>
#include <regex>

using namespace std;
using namespace std::chrono;

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


openai::Json call_openai(string prompt, int max_tokens, string model = "gpt-4-0125-preview") {
  auto openai_instance = openai::OpenAI();
    // cout << "Assembling request..." << endl;
    string request = R"({
       "model": ")" + model + R"(", 
	   "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": ")" + prompt + R"("}],
       "max_tokens": )" + to_string(max_tokens) + R"(,
       "temperature": 0.7
    })";
    // cout << request << endl;
    auto json_request = nlohmann::json::parse(request);

    // // cout << "Calling openai..." << endl;
    // auto completion = openai_instance.chat.create(json_request);

    for (int retry = 0; retry < 5; ++retry) {
    try {
        auto completion = openai_instance.chat.create(json_request);
        return completion;
    } catch (const std::exception& e) {
        // std::cerr << "API call failed (attempt " << retry + 1 << "): " << e.what();
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

    // // cout << "Returning completion..." << " " << completion["choices"][0]["message"]["content"] << endl;
    // return completion;
}


int serial(nlohmann::json formatted_prompts) {
    string original_prompt = formatted_prompts["original"];
    string escaped_prompt = escape_json(original_prompt);
    auto completion = call_openai(escaped_prompt, 4000);
    // cout << completion["choices"][0]["message"]["content"] << endl;
    return completion["usage"]["completion_tokens"];
}


// Function to obtain parallel schema
nlohmann::json generate_parallel_schema(nlohmann::json prompt) {
  string prompt_for_schema = R"(Parse the following question into the schema format. DO NOT provide answers to the questions. Only give the questions with no answers. Do not say anything else, beside returning the schema in the format described below. ENSURE that your response is in the format described below.'
  Schema:
  {
    "context": CONTEXT,
    "n": N
  }

  EXAMPLES (watch):
  Input: Generate 3 ideas for social media posts for a local bakery named NAME_1
  Schema:
  {
    "context": "Generate 1 idea for social media posts for a local bakery named NAME_1", 
    "n": 3
  }
  Input: I have a video talking about life being war. It"s a war to get money, women,  job, and friendship, literally everything in life is a war. Generate 5 different hooks giving an unpopular opinion that war is good . I want the hook to have ""NAME_1"" in it as the person who explains the video.
  {
    "context": "I have a video talking about life being war. It's a war to get money, women,  job, and friendship, literally everything in life is a war. Generate 1 hook giving an unpopular opinion that war is good . I want the hook to have ""NAME_1"" in it as the person who explains the video", 
    "n": 5
  }
  Input: generate 30 sentences with word  "captivating" with upper-intermediate lexis
  Schema:
  {
    "context": "generate 1 sentence with word  "captivating" with upper-intermediate lexis", 
    "n": 30
  }


  YOUR TURN:
  Input: )" + escape_json(prompt) + R"())";
  auto schema_result = call_openai(
      escape_json(prompt_for_schema), 1024, "gpt-4o-mini"
    )["choices"][0]["message"]["content"].get<std::string>();

  // cout << "Schema result: " << schema_result << endl;
  auto parsed_schema_result = nlohmann::json::parse(schema_result);
  string context = parsed_schema_result["context"];
  int n = parsed_schema_result["n"].get<int>();

  return {{"context", context}, {"n", n}};
}


int parallel(nlohmann::json formatted_prompts) {
  string original_prompt = formatted_prompts["original"];
	auto schema = generate_parallel_schema(original_prompt);
	int n = schema["n"].get<int>();
	string escaped_context = escape_json(schema["context"].get<string>());

    vector<future<openai::Json>> futures;
    for (int i = 0; i < n; ++i) {
      futures.push_back(async(launch::async, call_openai, escaped_context, 1000, "gpt-4-0125-preview"));
    }

    int parallel_tokens = 0;
    vector<openai::Json> results;
    for (auto & f : futures) {
      auto completion = f.get();
    	results.push_back(completion);
      parallel_tokens += completion["usage"]["completion_tokens"].get<int>();
    }

    // for (auto & completion : results) {
    //   cout << completion["choices"][0]["message"]["content"] << endl;
    //   cout << endl;
    // }

   return parallel_tokens;
}

int main() {
  nlohmann::json prompts_list = error_handling("prompts/generate_n_schemas.json");
  ofstream log_file("out/serial_vs_E2E_parallel.txt");

  double total_serial_duration = 0, total_parallel_duration = 0;
  int total_serial_tokens = 0, total_parallel_tokens = 0;
  int task_count = 0;

  for (auto& prompt : prompts_list) {
    if (task_count >= 10) break; // Only process the first 10

    log_file << "Prompt " << (task_count + 1) << ":" << endl;
    log_file << prompt.dump(4) << endl;

    // Serial execution
    auto start_serial = high_resolution_clock::now();
    int serial_tokens = serial(prompt);
    auto end_serial = high_resolution_clock::now();
    milliseconds serial_duration = duration_cast<milliseconds>(end_serial - start_serial);

    // parallel execution
    auto start_parallel = high_resolution_clock::now();
    int parallel_tokens = parallel(prompt);
    auto end_parallel = high_resolution_clock::now();
    milliseconds parallel_duration = duration_cast<milliseconds>(end_parallel - start_parallel);

    // Record results
    total_serial_duration += serial_duration.count();
    total_parallel_duration += parallel_duration.count();
    total_serial_tokens += serial_tokens;
    total_parallel_tokens += parallel_tokens;

    // Log results for this prompt
    log_file << "Serial duration: " << serial_duration.count() << " ms" << endl;
    log_file << "Serial tokens: " << serial_tokens << endl;
    log_file << "Parallel duration: " << parallel_duration.count() << " ms" << endl;
    log_file << "Parallel tokens: " << parallel_tokens << endl << endl;


    // Calculate speedup
    auto speedup = static_cast<double>(serial_duration.count()) / parallel_duration.count();
    log_file << "Speedup: " << speedup << "x" << endl;

    // Calculate normalized speedup for a fair comparison across tokens
    double normalized_speedup = (static_cast<double>(serial_duration.count()) / serial_tokens) /
                                (static_cast<double>(parallel_duration.count()) / parallel_tokens);
    log_file << "Normalized speedup: " << normalized_speedup << "x" << endl << endl;

    ++task_count;
  }

  // Calculate and print averages
  double avg_serial_duration = total_serial_duration / task_count;
  double avg_parallel_duration = total_parallel_duration / task_count;
  int avg_serial_tokens = total_serial_tokens / task_count;
  int avg_parallel_tokens = total_parallel_tokens / task_count;


  cout << endl << "Average Serial duration: " << avg_serial_duration << " ms" << endl;
  cout << "Average Parallel duration: " << avg_parallel_duration << " ms" << endl;
  cout << "Average Serial tokens: " << avg_serial_tokens << endl;
  cout << "Average Parallel tokens: " << avg_parallel_tokens << endl;

  // Save averages to log file
  log_file << "Average Serial duration: " << avg_serial_duration << " ms" << endl;
  log_file << "Average Parallel duration: " << avg_parallel_duration << " ms" << endl;
  log_file << "Average Serial tokens: " << avg_serial_tokens << endl;
  log_file << "Average Parallel tokens: " << avg_parallel_tokens << endl;

  // Calculate average speedup
  auto speedup = static_cast<double>(avg_serial_duration) / avg_parallel_duration;

  // Calculate normalized speedup for a fair comparison across tokens
  double normalized_speedup = (static_cast<double>(avg_serial_duration) / avg_serial_tokens) /
                              (static_cast<double>(avg_parallel_duration) / avg_parallel_tokens);

  cout << "Average Speedup: " << speedup << "x" << endl;
  cout << "Average Normalized speedup: " << normalized_speedup << "x" << endl;


  log_file << "Average Speedup: " << speedup << "x" << endl;
  log_file << "Average Normalized speedup: " << normalized_speedup << "x" << endl;

  log_file.close();
  return 0;
}
