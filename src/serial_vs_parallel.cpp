#include "openai.hpp"

#include <chrono>
#include <future>
#include <string>
#include <optional>
#include <regex>
#include <iostream>
#include <fstream>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <thread>
#include <unistd.h>
#include <getopt.h>

using namespace std;
using namespace std::chrono;

struct Schema {
    string original_prompt;
    string template_str;
    string context;
    vector<string> data_items;
    int n_count;
    string category;
    bool has_data;
    bool has_n;
    bool is_valid;
    
    Schema() : n_count(0), has_data(false), has_n(false), is_valid(false) {}
};

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

string getIthLetter(int i) {
    int idx = i % 26;
    char letter = 'A' + idx;
    return string(1, letter);
}

// Parse CSV row into fields
vector<string> parse_csv_row(const string& row) {
    vector<string> fields;
    stringstream ss(row);
    string field;
    bool in_quotes = false;
    string current_field = "";
    
    for (size_t i = 0; i < row.length(); ++i) {
        char c = row[i];
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(current_field);
            current_field = "";
        } else {
            current_field += c;
        }
    }
    fields.push_back(current_field); // Add last field
    return fields;
}

// Parse JSON array string into vector
vector<string> parse_json_array(const string& json_str) {
    vector<string> items;
    if (json_str.empty() || json_str == "\"\"") return items;
    
    try {
        auto json_array = nlohmann::json::parse(json_str);
        if (json_array.is_array()) {
            for (const auto& item : json_array) {
                items.push_back(item.get<string>());
            }
        }
    } catch (const exception& e) {
        // Skip invalid JSON arrays
    }
    return items;
}

// Validate schema
bool validate_schema(const Schema& schema) {
    // Must have either data OR n, never both
    if (schema.has_data && schema.has_n) return false;
    if (!schema.has_data && !schema.has_n) return false;
    
    // Must have template
    if (schema.template_str.empty()) return false;
    
    return true;
}

// Load and parse CSV file
vector<Schema> load_csv_schemas(const string& filepath) {
    vector<Schema> schemas;
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening CSV file: " << filepath << endl;
        return schemas;
    }
    
    string line;
    getline(file, line); // Skip header
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        auto fields = parse_csv_row(line);
        if (fields.size() < 12) continue; // Need at least the core fields
        
        Schema schema;
        
        // Parse CSV columns (based on your header)
        schema.original_prompt = fields[2];  // prompt
        schema.template_str = fields[8];     // template  
        schema.context = fields[9];          // context
        string data_str = fields[10];        // data
        string n_str = fields[11];           // n
        schema.category = fields[4];         // category
        
        // Parse data field
        if (!data_str.empty() && data_str != "\"\"") {
            schema.data_items = parse_json_array(data_str);
            schema.has_data = !schema.data_items.empty();
        }
        
        // Parse n field
        if (!n_str.empty() && n_str != "\"\"") {
            try {
                float n_float = stof(n_str);
                schema.n_count = static_cast<int>(n_float);
                schema.has_n = (schema.n_count > 0);
            } catch (const exception& e) {
                // Skip invalid n values
            }
        }
        
        // Validate schema
        schema.is_valid = validate_schema(schema);
        
        if (schema.is_valid) {
            schemas.push_back(schema);
        }
        // Skip invalid schemas silently
    }
    
    return schemas;
}

// Get optimized system prompt based on category
string get_system_prompt(const string& category) {
    if (category == "Keyword Extraction") {
        return "You are a helpful assistant specializing in keyword extraction. Only extract values for the given keyword and do not include any irrelevant information.";
    } else if (category == "Reading Comprehension") {
        return "You are a helpful assistant specializing in reading comprehension. Provide extremely concise and accurate answers based on the given context.";
    } else if (category == "Named Entity Recognition") {
        return "You are a helpful assistant specializing in named entity recognition. Extract entities accurately without irrelevant information.";
    } else if (category == "Language Correction") {
        return "You are a helpful assistant specializing in language correction. Correct errors while preserving original meaning; be concise.";
    } else if (category == "Translation") {
        return "You are a helpful assistant specializing in translation. Provide accurate translations without additional commentary.";
    } else if (category == "Sentiment Analysis") {
        return "You are a helpful assistant specializing in sentiment analysis. Provide concise sentiment classifications.";
    } else {
        return "You are a helpful assistant. Provide concise and accurate response based on the given task and context.";
    }
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
            if (retry < 4) {
                int delay = (1 << retry) * 1000; // Exponential backoff
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            } else {
                std::cerr << "Max retries reached. Failing: " << e.what() << endl;
                throw;
            }
        }
    }
    throw std::runtime_error("Unexpected error in call_openai");
}

openai::Json call_openai_postprocess(string system_prompt, string prompt, int max_tokens = 2000) {
    auto openai_instance = openai::OpenAI();
    string escaped_system = escape_json(system_prompt);
    string escaped_prompt = escape_json(prompt);
    string request = R"({
       "model": "gpt-4o-mini",
       "messages": [{"role": "system", "content": ")" + escaped_system + R"("}, {"role": "user", "content": ")" + escaped_prompt + R"("}],
       "max_tokens": )" + to_string(max_tokens) + R"(,
       "temperature": 0.3
    })";
    auto json_request = nlohmann::json::parse(request);

    for (int retry = 0; retry < 5; ++retry) {
        try {
            auto completion = openai_instance.chat.create(json_request);
            return completion;
        } catch (const std::exception& e) {
            if (retry < 4) {
                int delay = (1 << retry) * 1000;
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            } else {
                std::cerr << "Post-processing retry failed: " << e.what() << endl;
                throw;
            }
        }
    }
    throw std::runtime_error("Post-processing failed");
}

// Execute schema serially
pair<int, string> execute_serial(const Schema& schema) {
    string prompt = schema.original_prompt;
    string escaped_prompt = escape_json(prompt);
    string system_prompt = get_system_prompt(schema.category);
    
    auto completion = call_openai(system_prompt, escaped_prompt, 4000);
    string output = completion["choices"][0]["message"]["content"];
    return make_pair(completion["usage"]["completion_tokens"], output);
}

// Execute schema in parallel
tuple<vector<int>, int, vector<pair<int, openai::Json>>> execute_parallel(const Schema& schema) {
    string base_template = schema.template_str;
    string system_prompt = get_system_prompt(schema.category);
    int n_tasks = schema.has_data ? schema.data_items.size() : schema.n_count;
    
    vector<future<pair<long, openai::Json>>> futures;
    
    for (int i = 0; i < n_tasks; ++i) {
        string prompt = base_template;
        string task_system_prompt = system_prompt;
        
        // Replace context if present
        if (!schema.context.empty()) {
            string context_regex = "\\{context\\}";
            prompt = replace_in_string(prompt, context_regex, escape_json(schema.context));
        }
        
        if (schema.has_data) {
            // Replace {data} with current item
            string data_regex = "\\{data\\}";
            prompt = replace_in_string(prompt, data_regex, escape_json(schema.data_items[i]));
        } else if (schema.has_n) {
            // Replace {n} with "1" and add diversity
            string n_regex = "\\{n\\}";
            prompt = replace_in_string(prompt, n_regex, "1");
            task_system_prompt += " Try to make your response start with the letter " + getIthLetter(i);
        }
        
        string escaped_prompt = escape_json(prompt);
        
        futures.push_back(async(launch::async, [task_system_prompt, escaped_prompt]() {
            auto start = high_resolution_clock::now();
            auto completion = call_openai(task_system_prompt, escaped_prompt, 1000);
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

// Post-process parallel outputs to remove redundancy and improve flow
string post_process_outputs(const Schema& schema, const vector<string>& parallel_outputs) {
    string combined_outputs = "";
    for (size_t i = 0; i < parallel_outputs.size(); ++i) {
        combined_outputs += "Output " + to_string(i+1) + ": " + parallel_outputs[i] + "\n\n";
    }
    
    string post_process_prompt = "Original query: " + schema.original_prompt + "\n\n";
    post_process_prompt += "Parallel outputs:\n" + combined_outputs + "\n";
    post_process_prompt += "Combine these outputs into a single, coherent response. Remove redundant context, ensure smooth flow, and maintain all essential information.";
    
    string system_prompt = "You are a helpful assistant that combines and cleans up parallel outputs. Remove redundancy while preserving all key information and ensuring natural flow.";
    
    try {
        auto completion = call_openai_postprocess(system_prompt, post_process_prompt);
        return completion["choices"][0]["message"]["content"];
    } catch (const exception& e) {
        // If post-processing fails, return original parallel outputs
        return combined_outputs;
    }
}

int main(int argc, char* argv[]) {
    std::string queries;
    std::string output;
    std::string sample_size = "10";  // Default sample size
    bool enable_postprocessing = false;

    struct option long_options[] = {
        {"queries", required_argument, nullptr, 'q'},
        {"output", required_argument, nullptr, 'o'},
        {"sample-size", required_argument, nullptr, 's'},
        {"post-process", no_argument, nullptr, 'p'},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "q:o:s:p", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'q':
                queries = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            case 's':
                sample_size = optarg;
                break;
            case 'p':
                enable_postprocessing = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " --queries <csv_file> --output <output_file> [--sample-size <num|all>] [--post-process]" << std::endl;
                return 1;
        }
    }

    if (queries.empty() || output.empty()) {
        std::cerr << "Both --queries and --output are required." << std::endl;
        return 1;
    }

    std::cout << "Loading schemas from: " << queries << std::endl;
    
    vector<Schema> all_schemas = load_csv_schemas(queries);
    std::cout << "Loaded " << all_schemas.size() << " valid schemas" << std::endl;
    
    // Handle sample size
    vector<Schema> schemas;
    if (sample_size == "all") {
        schemas = all_schemas;
        std::cout << "Processing all " << schemas.size() << " schemas" << std::endl;
    } else {
        int sample_count = stoi(sample_size);
        sample_count = min(sample_count, static_cast<int>(all_schemas.size()));
        schemas.assign(all_schemas.begin(), all_schemas.begin() + sample_count);
        std::cout << "Processing sample of " << schemas.size() << " schemas (default sample size is 10). Set --sample-size to `all` to process entire dataset." << std::endl;
    }
    
    if (enable_postprocessing) {
        std::cout << "Post-processing enabled for cleaner outputs" << std::endl;
    }
    
    nlohmann::json results_json = nlohmann::json::array();
    
    double total_serial_duration = 0, total_parallel_duration = 0;
    int total_serial_tokens = 0, total_parallel_tokens = 0;
    int task_count = 0;

    for (const auto& schema : schemas) {
        nlohmann::json result_entry;
        result_entry["prompt"] = schema.original_prompt;
        result_entry["category"] = schema.category;

        cout << "Processing: " << schema.category << " - " << schema.original_prompt.substr(0, 50) << "..." << endl;

        // Serial execution
        auto start_serial = high_resolution_clock::now();
        auto [serial_tokens, serial_output] = execute_serial(schema);
        auto end_serial = high_resolution_clock::now();
        milliseconds serial_duration = duration_cast<milliseconds>(end_serial - start_serial);

        // Parallel execution
        auto start_parallel = high_resolution_clock::now();
        auto [parallel_tokens, sum_parallel_tokens, parallel_results] = execute_parallel(schema);
        auto end_parallel = high_resolution_clock::now();
        milliseconds parallel_duration = duration_cast<milliseconds>(end_parallel - start_parallel);

        // Record results
        total_serial_duration += serial_duration.count();
        total_parallel_duration += parallel_duration.count();
        total_serial_tokens += serial_tokens;
        total_parallel_tokens += sum_parallel_tokens;

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

        // Post-process if enabled
        if (enable_postprocessing) {
            string post_processed = post_process_outputs(schema, parallel_outputs);
            result_entry["post_processed_output"] = post_processed;
        }

        // Calculate speedup
        result_entry["speedup"] = static_cast<double>(serial_duration.count()) / parallel_duration.count();
        result_entry["normalized_speedup"] = (static_cast<double>(serial_duration.count()) / serial_tokens) / 
            (static_cast<double>(parallel_duration.count()) / sum_parallel_tokens);

        results_json.push_back(result_entry);
        ++task_count;
    }

    // Calculate averages
    nlohmann::json averages;
    averages["avg_serial_duration"] = total_serial_duration / task_count;
    averages["avg_parallel_duration"] = total_parallel_duration / task_count;
    averages["avg_serial_tokens"] = total_serial_tokens / task_count;
    averages["avg_parallel_tokens"] = total_parallel_tokens / task_count;

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