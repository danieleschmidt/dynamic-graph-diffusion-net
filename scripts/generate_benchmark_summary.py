#!/usr/bin/env python3
"""
Generate benchmark summary from pytest-benchmark JSON results.

This script processes benchmark results and generates human-readable summaries
for performance tracking and reporting.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class BenchmarkSummaryGenerator:
    """Generates benchmark summaries from pytest-benchmark results."""
    
    def __init__(self):
        self.results = {}
        self.summary_data = {}
    
    def load_benchmark_results(self, input_dir: str, benchmark: str, python_version: str):
        """Load benchmark results from JSON files."""
        input_path = Path(input_dir)
        
        # Look for the specific benchmark file
        pattern = f"{benchmark}_py{python_version}.json"
        result_files = list(input_path.glob(f"**/{pattern}"))
        
        if not result_files:
            print(f"No benchmark results found for {benchmark} (Python {python_version})")
            return
        
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    self.results[str(result_file)] = data
                    print(f"Loaded results from {result_file}")
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    def process_results(self, benchmark: str, python_version: str):
        """Process benchmark results and extract key metrics."""
        self.summary_data = {
            "benchmark_name": benchmark,
            "python_version": python_version,
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "statistics": {},
            "performance_insights": []
        }
        
        for file_path, data in self.results.items():
            benchmarks = data.get("benchmarks", [])
            
            for bench in benchmarks:
                test_info = {
                    "name": bench.get("name", "Unknown"),
                    "fullname": bench.get("fullname", ""),
                    "params": bench.get("params", {}),
                    "stats": bench.get("stats", {}),
                    "performance": self._extract_performance_metrics(bench)
                }
                self.summary_data["tests"].append(test_info)
        
        self._generate_statistics()
        self._generate_insights()
    
    def _extract_performance_metrics(self, benchmark_data: Dict) -> Dict:
        """Extract key performance metrics from benchmark data."""
        stats = benchmark_data.get("stats", {})
        
        return {
            "mean_time": stats.get("mean", 0),
            "min_time": stats.get("min", 0),
            "max_time": stats.get("max", 0),
            "stddev": stats.get("stddev", 0),
            "median": stats.get("median", 0),
            "iqr": stats.get("iqr", 0),
            "outliers": stats.get("outliers", "0;0"),
            "rounds": stats.get("rounds", 0)
        }
    
    def _generate_statistics(self):
        """Generate overall statistics from all tests."""
        if not self.summary_data["tests"]:
            return
        
        all_means = [test["performance"]["mean_time"] for test in self.summary_data["tests"]]
        all_stddevs = [test["performance"]["stddev"] for test in self.summary_data["tests"]]
        
        self.summary_data["statistics"] = {
            "total_tests": len(self.summary_data["tests"]),
            "fastest_test": min(all_means) if all_means else 0,
            "slowest_test": max(all_means) if all_means else 0,
            "average_time": sum(all_means) / len(all_means) if all_means else 0,
            "total_time": sum(all_means) if all_means else 0,
            "average_stddev": sum(all_stddevs) / len(all_stddevs) if all_stddevs else 0
        }
    
    def _generate_insights(self):
        """Generate performance insights and recommendations."""
        insights = []
        stats = self.summary_data["statistics"]
        
        if stats.get("total_tests", 0) == 0:
            insights.append("⚠️ No benchmark tests found")
            self.summary_data["performance_insights"] = insights
            return
        
        # Performance insights
        fastest = stats.get("fastest_test", 0)
        slowest = stats.get("slowest_test", 0)
        
        if slowest > 0 and fastest > 0:
            ratio = slowest / fastest
            if ratio > 100:
                insights.append(f"🐌 Large performance variance detected: {ratio:.1f}x difference between fastest and slowest tests")
            elif ratio > 10:
                insights.append(f"⚠️ Moderate performance variance: {ratio:.1f}x difference between fastest and slowest tests")
            else:
                insights.append(f"✅ Good performance consistency: {ratio:.1f}x difference between fastest and slowest tests")
        
        # Timing insights
        avg_time = stats.get("average_time", 0)
        if avg_time > 1.0:
            insights.append(f"🐌 Average test time is high: {avg_time:.3f}s - consider optimization")
        elif avg_time > 0.1:
            insights.append(f"⚠️ Average test time is moderate: {avg_time:.3f}s")
        else:
            insights.append(f"✅ Good average test time: {avg_time:.3f}s")
        
        # Stability insights
        avg_stddev = stats.get("average_stddev", 0)
        if avg_stddev > avg_time * 0.1:
            insights.append(f"⚠️ High variability in results: {avg_stddev:.3f}s stddev")
        else:
            insights.append(f"✅ Stable performance: {avg_stddev:.3f}s stddev")
        
        self.summary_data["performance_insights"] = insights
    
    def generate_markdown_summary(self, output_file: str):
        """Generate a markdown summary report."""
        benchmark_name = self.summary_data.get("benchmark_name", "Unknown")
        python_version = self.summary_data.get("python_version", "Unknown")
        timestamp = self.summary_data.get("timestamp", "Unknown")
        stats = self.summary_data.get("statistics", {})
        tests = self.summary_data.get("tests", [])
        insights = self.summary_data.get("performance_insights", [])
        
        markdown_content = f"""# 🏃 Benchmark Summary: {benchmark_name}

**Python Version:** {python_version}  
**Generated:** {timestamp}  
**Total Tests:** {stats.get('total_tests', 0)}

## 📊 Performance Overview

| Metric | Value |
|--------|-------|
| Total Tests | {stats.get('total_tests', 0)} |
| Fastest Test | {stats.get('fastest_test', 0):.6f}s |
| Slowest Test | {stats.get('slowest_test', 0):.6f}s |
| Average Time | {stats.get('average_time', 0):.6f}s |
| Total Time | {stats.get('total_time', 0):.6f}s |
| Average Std Dev | {stats.get('average_stddev', 0):.6f}s |

## 🔍 Performance Insights

"""
        
        for insight in insights:
            markdown_content += f"- {insight}\\n"
        
        if tests:
            markdown_content += f"""
## 🧪 Detailed Test Results

| Test Name | Mean Time (s) | Min Time (s) | Max Time (s) | Std Dev (s) | Rounds |
|-----------|---------------|--------------|--------------|-------------|---------|
"""
            
            for test in tests:
                perf = test["performance"]
                markdown_content += f"| {test['name']} | {perf['mean_time']:.6f} | {perf['min_time']:.6f} | {perf['max_time']:.6f} | {perf['stddev']:.6f} | {perf['rounds']} |\\n"
        
        markdown_content += f"""
## 📈 Performance Trends

*Note: Historical trend analysis requires multiple benchmark runs.*

## 🎯 Recommendations

Based on the benchmark results:

1. **Optimization Targets**: Focus on the slowest-performing tests
2. **Stability**: Investigate tests with high standard deviation
3. **Monitoring**: Set up alerts for performance regressions
4. **Baselines**: Use these results as performance baselines for future comparisons

## 🔗 Additional Resources

- [Benchmark Configuration](../pyproject.toml)
- [Performance Monitoring](../monitoring/README.md)
- [GitHub Actions Workflows](../.github/workflows/)

---

🤖 *This report was generated automatically by the DGDN benchmark system*
"""
        
        with open(output_file, 'w') as f:
            f.write(markdown_content)
        
        print(f"✅ Markdown summary saved to {output_file}")
    
    def generate_json_summary(self, output_file: str):
        """Generate a JSON summary report."""
        with open(output_file, 'w') as f:
            json.dump(self.summary_data, f, indent=2)
        
        print(f"✅ JSON summary saved to {output_file}")
    
    def generate_console_summary(self):
        """Print a console summary of the benchmark results."""
        benchmark_name = self.summary_data.get("benchmark_name", "Unknown")
        python_version = self.summary_data.get("python_version", "Unknown")
        stats = self.summary_data.get("statistics", {})
        insights = self.summary_data.get("performance_insights", [])
        
        print(f"\\n🏃 Benchmark Summary: {benchmark_name} (Python {python_version})")
        print("=" * 60)
        
        print(f"Total Tests: {stats.get('total_tests', 0)}")
        print(f"Fastest Test: {stats.get('fastest_test', 0):.6f}s")
        print(f"Slowest Test: {stats.get('slowest_test', 0):.6f}s")
        print(f"Average Time: {stats.get('average_time', 0):.6f}s")
        print(f"Total Time: {stats.get('total_time', 0):.6f}s")
        
        print("\\n🔍 Performance Insights:")
        for insight in insights:
            print(f"  {insight}")
        
        if self.summary_data.get("tests"):
            print("\\n🧪 Test Results:")
            for test in self.summary_data["tests"][:5]:  # Show top 5
                perf = test["performance"]
                print(f"  {test['name']}: {perf['mean_time']:.6f}s ± {perf['stddev']:.6f}s")
            
            if len(self.summary_data["tests"]) > 5:
                print(f"  ... and {len(self.summary_data['tests']) - 5} more tests")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate benchmark summary from results")
    parser.add_argument("--input-dir", required=True, help="Directory containing benchmark results")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--python-version", required=True, help="Python version")
    parser.add_argument("--output", help="Output file path (auto-generated if not specified)")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", help="Output format")
    parser.add_argument("--console", action="store_true", help="Also print console summary")
    
    args = parser.parse_args()
    
    generator = BenchmarkSummaryGenerator()
    generator.load_benchmark_results(args.input_dir, args.benchmark, args.python_version)
    generator.process_results(args.benchmark, args.python_version)
    
    if args.console:
        generator.generate_console_summary()
    
    # Generate output files
    if args.output:
        base_output = Path(args.output).stem
        output_dir = Path(args.output).parent
    else:
        base_output = f"summary_{args.benchmark}_py{args.python_version}"
        output_dir = Path(args.input_dir)
    
    if args.format in ["markdown", "both"]:
        md_output = output_dir / f"{base_output}.md"
        generator.generate_markdown_summary(str(md_output))
    
    if args.format in ["json", "both"]:
        json_output = output_dir / f"{base_output}.json"
        generator.generate_json_summary(str(json_output))
    
    print("🎉 Benchmark summary generation completed!")


if __name__ == "__main__":
    main()