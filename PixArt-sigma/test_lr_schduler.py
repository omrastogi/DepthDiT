def linear_decay_lr_schedule(current_step, warmup_steps, start_step, stop_step, base_lr, final_lr):
    """
    Learning rate schedule with warmup and linear decay to a specified final_lr, using ratio.

    Args:
        current_step (int): The current training step.
        warmup_steps (int): Number of steps for warmup.
        start_step (int): Step at which decay begins.
        stop_step (int): Step at which the scheduler stops and reaches final_lr.
        base_lr (float): The base learning rate (maximum value during training).
        final_lr (float): The final learning rate value at stop_step.

    Returns:
        float: The learning rate at the current step.
    """
    final_lr_ratio = final_lr / base_lr  # Ratio between final and base LR

    if warmup_steps > 0 and current_step < warmup_steps:
        # Warmup phase: linearly increase from 0 to base_lr
        lr = (current_step / warmup_steps)
    elif current_step < start_step:
        # After warmup but before decay begins: keep at base_lr
        lr = 1.0
    elif current_step >= stop_step:
        # After stop_step: maintain final_lr
        lr = final_lr_ratio
    else:
        # Linearly decay the learning rate multiplier from 1.0 to final_lr_ratio
        decay_steps = stop_step - start_step
        lr = 1.0 - ((current_step - start_step) / decay_steps) * (1.0 - final_lr_ratio)
    return lr


base_lr = 3e-4     # Starting learning rate
final_lr = 1e-5    # Final learning rate after decay
warmup_steps = 1000   # Number of steps to warm up
start_step = 1000    # Step at which decay starts
stop_step = 30000     # Step at which decay ends
total_steps = 40000   # Total number of training steps

import plotly.graph_objects as go
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

# Initialize a dummy optimizer
model = torch.nn.Linear(10, 1)  # Simple model
optimizer = SGD(model.parameters(), lr=base_lr)

# Define the LR scheduler using LambdaLR
lr_scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda step: linear_decay_lr_schedule(
        step, 
        warmup_steps=warmup_steps, 
        start_step=start_step, 
        stop_step=stop_step, 
        base_lr=base_lr, 
        final_lr=final_lr
    )
)

# Simulate the training process and collect learning rates
steps = list(range(total_steps))
learning_rates = []

for step in steps:
    optimizer.step()  # Dummy optimizer step
    lr_scheduler.step()  # Update the scheduler
    learning_rates.append(optimizer.param_groups[0]['lr'])  # Record the learning rate

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(steps, learning_rates, label="Learning Rate Schedule", linewidth=2)
plt.axvline(x=warmup_steps, color="orange", linestyle="--", label="Warmup Ends")
plt.axvline(x=start_step, color="green", linestyle="--", label="Decay Starts")
plt.axvline(x=stop_step, color="red", linestyle="--", label="Decay Ends")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule with Warmup and Linear Decay")
plt.legend()
plt.grid()
# Save the plot to a file
output_path = "learning_rate_schedule.png"
plt.savefig(output_path, dpi=300)
plt.close()
