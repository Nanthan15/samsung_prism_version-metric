import pandas as pd
from model import LeaderElectionModel
from kubernetes import client, config

# Load Kubernetes configuration
config.load_kube_config()
v1 = client.CoordinationV1Api()

def update_leader():
    # Load new metrics data from CSV
    data = pd.read_csv('simulated_pod_metrics.csv') #changes from pods.cdv to present
    data = data.dropna()  # Clean data

    # Initialize the model
    model = LeaderElectionModel()

    # Predict leader
    leaders = model.update_leader(data[['cpu_usage_cores', 'memory_usage_mib']])

    # Update Kubernetes lease
    for leader in leaders:
        print(leader)  # Log which pod is elected or not
        if 'Elected as leader' in leader:
            lease = {
                "metadata": {"name": "leader-lease"},
                "spec": {
                    "holderIdentity": leader.split()[1],  # Assuming format "Node X: ..."
                    "leaseDurationSeconds": 30
                }
            }
            v1.replace_namespaced_lease(name="leader-lease", namespace="default", body=lease)

update_leader()
