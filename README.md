# SafeRL Manipulator

**Risk-Aware Residual Reinforcement Learning for Autonomous Robotic Manipulation Using Large-Scale Robot Demonstrations**

This project combines Reinforcement Learning, Robotics simulation, Python, and the large-scale DROID robot dataset to create a safe manipulation policy.

## Architecture Diagram
```text
                    DROID
          Large-Scale Robot Dataset
                       |
                       v
              Data Streaming Layer
                       |
                       v
              State/Action Adapter
                       |
                       v
              Behavior Cloning
              PyTorch Neural Network
                       |
                       v
                  BC Policy
                  Action Prior
                       |
                       |
        +--------------+--------------+
        |                             |
        v                             v
   Robot State                 PPO Residual
        |                             |
        +--------------+--------------+
                       |
                       v
                Safe Environment
                       |
            +----------+----------+
            |          |          |
            v          v          v
          Panda      Cube      Obstacles
            |
            v
      Physics Simulation
            |
            v
          Reward
            |
     +------+-------+
     |              |
     v              v
 Safety          Efficiency
collision        energy/
clearance        smoothness
     |              |
     +------+-------+
            |
            v
       PPO Update
            |
            v
     Improved Robot Policy
```

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python scripts/setup_check.py
```

### 3. Inspect Dataset

```bash
python scripts/inspect_droid.py
```

### 4. Train Behavior Cloning Prior

```bash
python scripts/train_bc.py --max-samples 50000 --epochs 5
```

### 5. Train RL (Scratch, Residual, or Safe)

```bash
python scripts/train_rl.py --mode safe --timesteps 10000
```

### 6. Evaluate and Compare

```bash
python scripts/evaluate.py --method safe --episodes 10
python scripts/evaluate.py --all-methods
```

### 7. Run Dashboard

```bash
streamlit run dashboard/app.py
```

### Smoke Test (All-in-one)

```bash
python scripts/run_all.py --smoke-test
```

## Limitations
- This is a simulation-based research/engineering project.
- The final policy has not been validated on a physical robot.
- The energy metric is an action-magnitude proxy.
- Simulation physics differ from real-world dynamics.
