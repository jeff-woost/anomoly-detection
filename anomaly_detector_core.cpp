// anomaly_detector_core.cpp
// High-performance C++ core for trade anomaly detection

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

namespace py = pybind11;

// Trade structure
struct Trade {
    std::string trade_id;
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
    std::string side;
    double quantity;
    double price;
    double commission;
    std::string trader_id;
    std::string strategy;
    std::string venue;
    double latency_ms;
    double slippage;
    double market_impact;
};

// PnL structure
struct PnL {
    std::chrono::system_clock::time_point timestamp;
    std::string trader_id;
    std::string strategy;
    std::string symbol;
    double realized_pnl;
    double unrealized_pnl;
    double total_pnl;
    double position_size;
    double vwap;
    double sharpe_ratio;
    double max_drawdown;
    double win_rate;
};

// Anomaly structure
struct Anomaly {
    std::string anomaly_id;
    std::chrono::system_clock::time_point timestamp;
    std::string anomaly_type;
    int severity;
    double confidence;
    std::string affected_entity;
    std::string description;
    std::unordered_map<std::string, double> metrics;
    std::string recommended_action;
    std::string ml_model_used;
    bool false_positive;
};

// Fast statistics calculator
class FastStats {
private:
    std::vector<double> data;
    mutable bool stats_calculated;
    mutable double mean_val;
    mutable double std_val;
    mutable double median_val;
    mutable std::mutex stats_mutex;

    void calculate_stats() const {
        if (stats_calculated || data.empty()) return;
        
        std::lock_guard<std::mutex> lock(stats_mutex);
        if (stats_calculated) return;
        
        // Calculate mean
        mean_val = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
        
        // Calculate standard deviation
        double sq_sum = 0.0;
        for (const auto& val : data) {
            sq_sum += (val - mean_val) * (val - mean_val);
        }
        std_val = std::sqrt(sq_sum / data.size());
        
        // Calculate median
        std::vector<double> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        size_t n = sorted_data.size();
        median_val = (n % 2 == 0) ? 
            (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2.0 : 
            sorted_data[n/2];
        
        stats_calculated = true;
    }

public:
    FastStats() : stats_calculated(false), mean_val(0), std_val(0), median_val(0) {}
    
    void add_value(double val) {
        data.push_back(val);
        stats_calculated = false;
    }
    
    void add_values(const std::vector<double>& vals) {
        data.insert(data.end(), vals.begin(), vals.end());
        stats_calculated = false;
    }
    
    double mean() const {
        calculate_stats();
        return mean_val;
    }
    
    double std_dev() const {
        calculate_stats();
        return std_val;
    }
    
    double median() const {
        calculate_stats();
        return median_val;
    }
    
    double percentile(double p) const {
        if (data.empty()) return 0.0;
        
        std::vector<double> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        
        size_t index = static_cast<size_t>(p * sorted_data.size() / 100.0);
        return sorted_data[std::min(index, sorted_data.size() - 1)];
    }
    
    double z_score(double value) const {
        calculate_stats();
        return (std_val > 0) ? (value - mean_val) / std_val : 0.0;
    }
    
    double mad() const {  // Median Absolute Deviation
        if (data.empty()) return 0.0;
        
        double med = median();
        std::vector<double> deviations;
        
        for (const auto& val : data) {
            deviations.push_back(std::abs(val - med));
        }
        
        FastStats dev_stats;
        dev_stats.add_values(deviations);
        return dev_stats.median();
    }
    
    void clear() {
        data.clear();
        stats_calculated = false;
    }
    
    size_t size() const { return data.size(); }
};

// High-performance anomaly detector
class HighPerformanceAnomalyDetector {
private:
    std::unordered_map<std::string, FastStats> symbol_volume_stats;
    std::unordered_map<std::string, FastStats> trader_pnl_stats;
    std::unordered_map<std::string, FastStats> latency_stats;
    std::mutex stats_mutex;
    
    // Configuration
    double volume_threshold_multiplier = 3.0;
    double pnl_zscore_threshold = 3.0;
    double latency_percentile_threshold = 99.0;
    double concentration_threshold = 0.25;
    
    // Thread pool for parallel processing
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop = false;

public:
    HighPerformanceAnomalyDetector(size_t num_threads = 4) {
        // Initialize thread pool
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    ~HighPerformanceAnomalyDetector() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }
    
    // Update statistics with new trades
    void update_trade_stats(const std::vector<Trade>& trades) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        // Update volume statistics by symbol
        std::unordered_map<std::string, double> symbol_volumes;
        for (const auto& trade : trades) {
            symbol_volumes[trade.symbol] += trade.quantity;
        }
        
        for (const auto& [symbol, volume] : symbol_volumes) {
            symbol_volume_stats[symbol].add_value(volume);
        }
        
        // Update latency statistics
        for (const auto& trade : trades) {
            latency_stats[trade.venue].add_value(trade.latency_ms);
        }
    }
    
    // Update statistics with new PnL data
    void update_pnl_stats(const std::vector<PnL>& pnl_data) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        for (const auto& pnl : pnl_data) {
            trader_pnl_stats[pnl.trader_id].add_value(pnl.total_pnl);
        }
    }
    
    // Fast volume anomaly detection
    std::vector<Anomaly> detect_volume_anomalies(const std::vector<Trade>& trades) {
        std::vector<Anomaly> anomalies;
        
        // Group trades by symbol
        std::unordered_map<std::string, double> current_volumes;
        for (const auto& trade : trades) {
            current_volumes[trade.symbol] += trade.quantity;
        }
        
        // Check each symbol
        for (const auto& [symbol, volume] : current_volumes) {
            auto it = symbol_volume_stats.find(symbol);
            if (it != symbol_volume_stats.end() && it->second.size() > 10) {
                double z_score = it->second.z_score(volume);
                
                if (std::abs(z_score) > volume_threshold_multiplier) {
                    Anomaly anomaly;
                    anomaly.anomaly_id = "VOL_" + std::to_string(
                        std::chrono::system_clock::now().time_since_epoch().count());
                    anomaly.timestamp = std::chrono::system_clock::now();
                    anomaly.anomaly_type = "UNUSUAL_VOLUME";
                    anomaly.severity = (std::abs(z_score) > 5.0) ? 3 : 2;
                    anomaly.confidence = std::min(std::abs(z_score) / 5.0, 1.0);
                    anomaly.affected_entity = symbol;
                    anomaly.description = "Unusual trading volume detected for " + symbol;
                    anomaly.metrics["volume"] = volume;
                    anomaly.metrics["z_score"] = z_score;
                    anomaly.metrics["mean_volume"] = it->second.mean();
                    anomaly.metrics["std_volume"] = it->second.std_dev();
                    anomaly.recommended_action = "Review market conditions and trading algorithms";
                    anomaly.ml_model_used = "Statistical Z-Score (C++)";
                    anomaly.false_positive = false;
                    
                    anomalies.push_back(anomaly);
                }
            }
        }
        
        return anomalies;
    }
    
    // Fast PnL anomaly detection
    std::vector<Anomaly> detect_pnl_anomalies(const std::vector<PnL>& pnl_data) {
        std::vector<Anomaly> anomalies;
        
        for (const auto& pnl : pnl_data) {
            auto it = trader_pnl_stats.find(pnl.trader_id);
            if (it != trader_pnl_stats.end() && it->second.size() > 20) {
                double z_score = it->second.z_score(pnl.total_pnl);
                
                if (std::abs(z_score) > pnl_zscore_threshold) {
                    Anomaly anomaly;
                    anomaly.anomaly_id = "PNL_" + std::to_string(
                        std::chrono::system_clock::now().time_since_epoch().count());
                    anomaly.timestamp = std::chrono::system_clock::now();
                    anomaly.anomaly_type = "EXTREME_PNL";
                    anomaly.severity = (std::abs(z_score) > 4.0) ? 4 : 3;
                    anomaly.confidence = std::min(std::abs(z_score) / 4.0, 1.0);
                    anomaly.affected_entity = pnl.trader_id;
                    anomaly.description = "Extreme P&L movement detected for " + pnl.trader_id;
                    anomaly.metrics["pnl"] = pnl.total_pnl;
                    anomaly.metrics["z_score"] = z_score;
                    anomaly.metrics["mean_pnl"] = it->second.mean();
                    anomaly.metrics["std_pnl"] = it->second.std_dev();
                    anomaly.metrics["sharpe_ratio"] = pnl.sharpe_ratio;
                    anomaly.metrics["max_drawdown"] = pnl.max_drawdown;
                    anomaly.recommended_action = "Investigate positions and market movements";
                    anomaly.ml_model_used = "Statistical Z-Score (C++)";
                    anomaly.false_positive = false;
                    
                    anomalies.push_back(anomaly);
                }
            }
        }
        
        return anomalies;
    }
    
    // Fast latency anomaly detection
    std::vector<Anomaly> detect_latency_anomalies(const std::vector<Trade>& trades) {
        std::vector<Anomaly> anomalies;
        
        for (const auto& trade : trades) {
            auto it = latency_stats.find(trade.venue);
            if (it != latency_stats.end() && it->second.size() > 100) {
                double percentile_99 = it->second.percentile(latency_percentile_threshold);
                
                if (trade.latency_ms > percentile_99) {
                    Anomaly anomaly;
                    anomaly.anomaly_id = "LAT_" + trade.trade_id;
                    anomaly.timestamp = trade.timestamp;
                    anomaly.anomaly_type = "LATENCY_SPIKE";
                    anomaly.severity = (trade.latency_ms > percentile_99 * 2) ? 3 : 2;
                    anomaly.confidence = 0.9;
                    anomaly.affected_entity = trade.venue;
                    anomaly.description = "Execution latency spike detected";
                    anomaly.metrics["latency_ms"] = trade.latency_ms;
                    anomaly.metrics["p99_latency"] = percentile_99;
                    anomaly.metrics["median_latency"] = it->second.median();
                    anomaly.recommended_action = "Check network and system performance";
                    anomaly.ml_model_used = "Percentile Analysis (C++)";
                    anomaly.false_positive = false;
                    
                    anomalies.push_back(anomaly);
                }
            }
        }
        
        return anomalies;
    }
    
    // Concentration risk detection
    std::vector<Anomaly> detect_concentration_risk(const std::vector<Trade>& trades) {
        std::vector<Anomaly> anomalies;
        
        // Calculate exposure by symbol
        std::unordered_map<std::string, double> symbol_exposure;
        double total_exposure = 0.0;
        
        for (const auto& trade : trades) {
            double exposure = trade.quantity * trade.price;
            symbol_exposure[trade.symbol] += exposure;
            total_exposure += exposure;
        }
        
        // Check concentration
        for (const auto& [symbol, exposure] : symbol_exposure) {
            double concentration = exposure / total_exposure;
            
            if (concentration > concentration_threshold) {
                Anomaly anomaly;
                anomaly.anomaly_id = "CONC_" + symbol + "_" + 
                    std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
                anomaly.timestamp = std::chrono::system_clock::now();
                anomaly.anomaly_type = "CONCENTRATION_RISK";
                anomaly.severity = (concentration > 0.4) ? 4 : 3;
                anomaly.confidence = 0.95;
                anomaly.affected_entity = symbol;
                anomaly.description = "High position concentration detected";
                anomaly.metrics["concentration"] = concentration;
                anomaly.metrics["exposure"] = exposure;
                anomaly.metrics["total_exposure"] = total_exposure;
                anomaly.recommended_action = "Consider position reduction for risk management";
                anomaly.ml_model_used = "Concentration Analysis (C++)";
                anomaly.false_positive = false;
                
                anomalies.push_back(anomaly);
            }
        }
        
        return anomalies;
    }
    
    // Main detection function
    std::vector<Anomaly> detect_all_anomalies(const std::vector<Trade>& trades,
                                              const std::vector<PnL>& pnl_data) {
        std::vector<Anomaly> all_anomalies;
        std::vector<std::future<std::vector<Anomaly>>> futures;
        
        // Update statistics
        update_trade_stats(trades);
        update_pnl_stats(pnl_data);
        
        // Run detections in parallel
        auto volume_future = std::async(std::launch::async, 
            [this, &trades]() { return detect_volume_anomalies(trades); });
        
        auto pnl_future = std::async(std::launch::async,
            [this, &pnl_data]() { return detect_pnl_anomalies(pnl_data); });
        
        auto latency_future = std::async(std::launch::async,
            [this, &trades]() { return detect_latency_anomalies(trades); });
        
        auto concentration_future = std::async(std::launch::async,
            [this, &trades]() { return detect_concentration_risk(trades); });
        
        // Collect results
        auto volume_anomalies = volume_future.get();
        auto pnl_anomalies = pnl_future.get();
        auto latency_anomalies = latency_future.get();
        auto concentration_anomalies = concentration_future.get();
        
        // Combine all anomalies
        all_anomalies.insert(all_anomalies.end(), 
            volume_anomalies.begin(), volume_anomalies.end());
        all_anomalies.insert(all_anomalies.end(), 
            pnl_anomalies.begin(), pnl_anomalies.end());
        all_anomalies.insert(all_anomalies.end(), 
            latency_anomalies.begin(), latency_anomalies.end());
        all_anomalies.insert(all_anomalies.end(), 
            concentration_anomalies.begin(), concentration_anomalies.end());
        
        return all_anomalies;
    }
    
    // Get current statistics
    py::dict get_statistics() {
        py::dict stats;
        
        py::dict symbol_stats;
        for (const auto& [symbol, stat] : symbol_volume_stats) {
            py::dict s;
            s["mean"] = stat.mean();
            s["std"] = stat.std_dev();
            s["median"] = stat.median();
            s["count"] = stat.size();
            symbol_stats[symbol] = s;
        }
        stats["symbol_volume_stats"] = symbol_stats;
        
        py::dict trader_stats;
        for (const auto& [trader, stat] : trader_pnl_stats) {
            py::dict s;
            s["mean"] = stat.mean();
            s["std"] = stat.std_dev();
            s["median"] = stat.median();
            s["count"] = stat.size();
            trader_stats[trader] = s;
        }
        stats["trader_pnl_stats"] = trader_stats;
        
        return stats;
    }
};

// Python bindings
PYBIND11_MODULE(anomaly_detector_core, m) {
    m.doc() = "High-performance anomaly detection core for trading systems";
    
    // Trade class
    py::class_<Trade>(m, "Trade")
        .def(py::init<>())
        .def_readwrite("trade_id", &Trade::trade_id)
        .def_readwrite("symbol", &Trade::symbol)
        .def_readwrite("side", &Trade::side)
        .def_readwrite("quantity", &Trade::quantity)
        .def_readwrite("price", &Trade::price)
        .def_readwrite("commission", &Trade::commission)
        .def_readwrite("trader_id", &Trade::trader_id)
        .def_readwrite("strategy", &Trade::strategy)
        .def_readwrite("venue", &Trade::venue)
        .def_readwrite("latency_ms", &Trade::latency_ms)
        .def_readwrite("slippage", &Trade::slippage)
        .def_readwrite("market_impact", &Trade::market_impact);
    
    // PnL class
    py::class_<PnL>(m, "PnL")
        .def(py::init<>())
        .def_readwrite("trader_id", &PnL::trader_id)
        .def_readwrite("strategy", &PnL::strategy)
        .def_readwrite("symbol", &PnL::symbol)
        .def_readwrite("realized_pnl", &PnL::realized_pnl)
        .def_readwrite("unrealized_pnl", &PnL::unrealized_pnl)
        .def_readwrite("total_pnl", &PnL::total_pnl)
        .def_readwrite("position_size", &PnL::position_size)
        .def_readwrite("vwap", &PnL::vwap)
        .def_readwrite("sharpe_ratio", &PnL::sharpe_ratio)
        .def_readwrite("max_drawdown", &PnL::max_drawdown)
        .def_readwrite("win_rate", &PnL::win_rate);
    