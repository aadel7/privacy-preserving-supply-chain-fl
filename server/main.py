import flwr as fl

def weighted_average(metrics):
    # Calculate weighted average for accuracy and f1_score based on client dataset sizes
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "federated_accuracy": sum(accuracies) / sum(examples),
        "federated_f1_score": sum(f1_scores) / sum(examples),
    }

def main():
    print("Starting Federated Learning Central Server...")
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()