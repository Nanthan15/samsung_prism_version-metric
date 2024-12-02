#used to collect metrix of real time pods.. 

import subprocess
import csv
import re
from datetime import datetime, timezone

def collect_pod_metrics(namespace='default'):
    """Collects pod metrics (CPU, Memory, etc.) and stores them in a CSV."""
    cmd = f"kubectl top pods --namespace={namespace} --no-headers"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error collecting pod metrics")
        print(result.stderr)
        return

    pod_metrics = []
    current_time = datetime.now(timezone.utc)  # Make current_time offset-aware

    for line in result.stdout.strip().split("\n"):
        fields = line.split()
        pod_name = fields[0]
        cpu_usage = fields[1].replace('m', '')  # Remove 'm' for millicores
        memory_usage = fields[2].replace('Mi', '')  # Remove 'Mi' for memory units

        network_latency = get_network_latency(pod_name)
        uptime = get_pod_uptime(pod_name, namespace, current_time)

        pod_metrics.append({
            'pod_name': pod_name,
            'cpu_usage': int(cpu_usage),
            'memory_usage': int(memory_usage),
            'network_latency': network_latency,
            'uptime': uptime
        })

    with open('pod_metrics.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['pod_name', 'cpu_usage', 'memory_usage', 'network_latency', 'uptime'])
        writer.writeheader()
        writer.writerows(pod_metrics)

    print("Pod metrics collected and saved to pod_metrics.csv")

def get_network_latency(pod_name):
    """Calculates the network latency for a pod by pinging the node."""
    cmd = f"ping -c 1 {pod_name}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        match = re.search(r'time=(\d+\.\d+) ms', result.stdout)
        if match:
            return float(match.group(1))
    return 0  # Return 0 if latency could not be determined

def get_pod_uptime(pod_name, namespace, current_time):
    """Calculates pod uptime by parsing its start time."""
    cmd = f"kubectl describe pod {pod_name} -n {namespace}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error describing pod {pod_name}")
        return 0

    match = re.search(r'Start Time:\s+(\w{3}, \d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2} [+-]\d{4})', result.stdout)
    if match:
        start_time_str = match.group(1).strip()
        try:
            start_time = datetime.strptime(start_time_str, "%a, %d %b %Y %H:%M:%S %z")
            uptime = (current_time - start_time).total_seconds()
            return int(uptime)
        except ValueError as e:
            print(f"Error parsing start time for pod {pod_name}: {e}")
            return 0

    return 0

if __name__ == "__main__":
    collect_pod_metrics(namespace='default')
