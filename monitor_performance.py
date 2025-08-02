#!/usr/bin/env python3
"""
AgriLens AI Performance Monitor
Monitors system resources and application performance in real-time.
"""

import time
import psutil
import json
import os
from datetime import datetime
from pathlib import Path
import argparse

class PerformanceMonitor:
    def __init__(self, log_file="logs/performance.log", interval=5):
        self.log_file = Path(log_file)
        self.interval = interval
        self.log_file.parent.mkdir(exist_ok=True)
        
    def get_system_info(self):
        """Get current system information."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('.')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # GPU info (if available)
            gpu_info = self.get_gpu_info()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'gpu': gpu_info
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_gpu_info(self):
        """Get GPU information if available."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        'name': props.name,
                        'memory_total': props.total_memory,
                        'memory_allocated': torch.cuda.memory_allocated(i),
                        'memory_cached': torch.cuda.memory_reserved(i)
                    })
                
                return gpu_info
            else:
                return None
        except ImportError:
            return None
    
    def format_size(self, bytes):
        """Format bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"
    
    def print_status(self, info):
        """Print current status to console."""
        print(f"\n{'='*60}")
        print(f"🌱 AgriLens AI Performance Monitor - {info['timestamp']}")
        print(f"{'='*60}")
        
        # CPU
        print(f"🖥️  CPU: {info['cpu']['percent']:.1f}% ({info['cpu']['count']} cores)")
        
        # Memory
        mem = info['memory']
        print(f"💾 Memory: {self.format_size(mem['used'])} / {self.format_size(mem['total'])} ({mem['percent']:.1f}%)")
        
        # Disk
        disk = info['disk']
        print(f"💿 Disk: {self.format_size(disk['used'])} / {self.format_size(disk['total'])} ({disk['percent']:.1f}%)")
        
        # Network
        net = info['network']
        print(f"🌐 Network: ↑ {self.format_size(net['bytes_sent'])} ↓ {self.format_size(net['bytes_recv'])}")
        
        # GPU
        if info['gpu']:
            for i, gpu in enumerate(info['gpu']):
                mem_alloc = self.format_size(gpu['memory_allocated'])
                mem_total = self.format_size(gpu['memory_total'])
                print(f"🎮 GPU {i}: {gpu['name']} - {mem_alloc} / {mem_total}")
        
        print(f"{'='*60}")
    
    def log_info(self, info):
        """Log information to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(info) + '\n')
        except Exception as e:
            print(f"Error logging to file: {e}")
    
    def generate_report(self, duration_minutes=60):
        """Generate a performance report from logs."""
        print(f"\n📊 Generating performance report for last {duration_minutes} minutes...")
        
        if not self.log_file.exists():
            print("No log file found.")
            return
        
        # Read recent logs
        cutoff_time = datetime.now().timestamp() - (duration_minutes * 60)
        recent_logs = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    log_time = datetime.fromisoformat(data['timestamp']).timestamp()
                    if log_time > cutoff_time:
                        recent_logs.append(data)
                except:
                    continue
        
        if not recent_logs:
            print("No recent data found.")
            return
        
        # Calculate averages
        cpu_avg = sum(log['cpu']['percent'] for log in recent_logs) / len(recent_logs)
        memory_avg = sum(log['memory']['percent'] for log in recent_logs) / len(recent_logs)
        
        print(f"\n📈 Performance Summary ({len(recent_logs)} samples):")
        print(f"   CPU Usage (avg): {cpu_avg:.1f}%")
        print(f"   Memory Usage (avg): {memory_avg:.1f}%")
        
        # Find peak usage
        cpu_peak = max(log['cpu']['percent'] for log in recent_logs)
        memory_peak = max(log['memory']['percent'] for log in recent_logs)
        
        print(f"   CPU Usage (peak): {cpu_peak:.1f}%")
        print(f"   Memory Usage (peak): {memory_peak:.1f}%")
    
    def monitor(self, duration=None):
        """Start monitoring."""
        print("🌱 Starting AgriLens AI Performance Monitor...")
        print(f"📝 Logging to: {self.log_file}")
        print(f"⏱️  Interval: {self.interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        start_time = time.time()
        sample_count = 0
        
        try:
            while True:
                info = self.get_system_info()
                self.print_status(info)
                self.log_info(info)
                
                sample_count += 1
                
                # Check if duration limit reached
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n⏰ Monitoring completed after {duration} seconds")
                    break
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")
            print(f"📊 Total samples collected: {sample_count}")
        
        # Generate summary
        if sample_count > 0:
            self.generate_report()

def main():
    parser = argparse.ArgumentParser(description="AgriLens AI Performance Monitor")
    parser.add_argument("--interval", "-i", type=int, default=5, 
                       help="Monitoring interval in seconds (default: 5)")
    parser.add_argument("--duration", "-d", type=int, 
                       help="Monitoring duration in seconds (default: infinite)")
    parser.add_argument("--log-file", "-l", default="logs/performance.log",
                       help="Log file path (default: logs/performance.log)")
    parser.add_argument("--report", "-r", action="store_true",
                       help="Generate report from existing logs")
    parser.add_argument("--report-duration", "-rd", type=int, default=60,
                       help="Report duration in minutes (default: 60)")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.log_file, args.interval)
    
    if args.report:
        monitor.generate_report(args.report_duration)
    else:
        monitor.monitor(args.duration)

if __name__ == "__main__":
    main() 