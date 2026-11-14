import numpy as np
import json
from pathlib import Path

def dirichlet_partition(
    labels,
    num_clients=5,
    alpha=0.3,
    seed=42
):
    """
    Partition dataset indices among clients using
    class-wise Dirichlet allocation.

    Args:
        labels: Original dataset labels.
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter.
        seed: Random seed for reproducibility.

    Returns:
        client_indices: Dictionary mapping client IDs
                        to their assigned dataset indices.
    """

    if num_clients < 1:
        raise ValueError("num_clients must be positive.")

    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    labels = np.asarray(labels).reshape(-1)

    rng = np.random.default_rng(seed)

    classes = np.unique(labels)

    client_indices = {
        client_id: []
        for client_id in range(num_clients)
    }

    # Process each class independently.
    for class_id in classes:

        # Find indices belonging to this class.
        class_indices = np.where(labels == class_id)[0]

        # Shuffle indices to avoid ordering bias.
        rng.shuffle(class_indices)

        # Sample class allocation probabilities.
        proportions = rng.dirichlet(
            np.repeat(alpha, num_clients)
        )

        # Convert probabilities into integer sample counts.
        client_counts = rng.multinomial(
            len(class_indices),
            proportions
        )

        # Allocate class samples to clients.
        start = 0

        for client_id, count in enumerate(client_counts):

            end = start + count

            assigned_indices = class_indices[start:end]

            client_indices[client_id].extend(
                assigned_indices.tolist()
            )

            start = end

    # Shuffle each client's complete dataset.
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

        client_indices[client_id] = np.asarray(
            client_indices[client_id],
            dtype=np.int64
        )

    return client_indices


def validate_partition(
    client_indices,
    total_samples,
    num_clients
):
    """
    Verify that every sample is assigned exactly once.
    """

    assert len(client_indices) == num_clients

    all_indices = np.concatenate([
        client_indices[client_id]
        for client_id in range(num_clients)
    ])

    # Check total number of allocated samples.
    assert len(all_indices) == total_samples, (
        "Incorrect total number of allocated samples."
    )

    # Check duplicate assignments.
    assert len(np.unique(all_indices)) == total_samples, (
        "Duplicate sample assignments detected."
    )

    # Check that every index is within the dataset.
    assert all_indices.min() == 0
    assert all_indices.max() == total_samples - 1

    print("\nPartition validation successful.")
    print(f"Total samples : {total_samples}")
    print(f"Assigned      : {len(all_indices)}")
    print(f"Unique indices: {len(np.unique(all_indices))}")


def print_client_distribution(
    client_indices,
    labels,
    num_clients,
    num_classes=9
):
    """
    Display client sizes and class distributions.
    """

    labels = np.asarray(labels).reshape(-1)

    print("\nCLIENT DATA DISTRIBUTION")
    print("=" * 65)

    for client_id in range(num_clients):

        indices = client_indices[client_id]

        client_labels = labels[indices]

        class_counts = np.bincount(
            client_labels.astype(int),
            minlength=num_classes
        )

        print(f"\nClient {client_id}")
        print(f"Total samples: {len(indices)}")
        print(f"Class distribution: {class_counts.tolist()}")

def save_partition(
    client_indices,
    output_path,
    alpha,
    seed
):
    """Save client partition indices and configuration."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    partition_data = {
        "num_clients": len(client_indices),
        "alpha": alpha,
        "seed": seed,
        "client_indices": {
            str(client_id): indices.tolist()
            for client_id, indices in client_indices.items()
        }
    }

    with open(output_path, "w") as file:
        json.dump(partition_data, file)

    print(f"\nPartition saved to: {output_path}")