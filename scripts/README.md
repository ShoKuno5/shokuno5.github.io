# Blog Research Scripts

This directory contains automated research and improvement scripts for the ML blog.

## Scripts Overview

### 🚀 Main Scripts
- **`parallel-research.sh`** - Full parallel deep research with real-time dashboard (3 AI agents)
- **`parallel-simple.sh`** - Basic parallel research (simplified version)
- **`nightly-enhanced.sh`** - Enhanced nightly improvements with multiple agents
- **`nightly-deep.sh`** - Deep nightly research analysis
- **`nightly-basic.sh`** - Basic nightly improvements

### 🎯 Recommended Usage

**Use the main controller instead of calling scripts directly:**

```bash
# From project root
./research parallel    # Full parallel research with dashboard
./research simple      # Basic parallel research  
./research nightly     # Nightly improvements
./research schedule 02:00  # Schedule at 2 AM daily
./research status      # Check what's running
./research logs        # View recent logs
./research clean       # Clean up old files
```

### 📊 Script Comparison

| Script | Agents | Dashboard | Duration | Use Case |
|--------|--------|-----------|----------|----------|
| `parallel-research.sh` | 3 (Content, Citations, Code) | ✅ Real-time | ~10-30 min | **Recommended** - Full analysis |
| `parallel-simple.sh` | 3 | ❌ Basic output | ~5-15 min | Quick research |
| `nightly-enhanced.sh` | 4+ | ❌ | ~20-60 min | Scheduled improvements |
| `nightly-deep.sh` | 1 | ❌ | ~30+ min | Deep single analysis |
| `nightly-basic.sh` | 4 | ❌ | ~15-30 min | Basic scheduled |

### 🔧 Technical Details

- **Authentication**: Uses Claude Max plan session (no extra billing)
- **Parallel Processing**: Multiple Claude agents run simultaneously  
- **Logging**: All output saved to `logs/` directory
- **Git Integration**: Creates branches and optionally PRs
- **System Resources**: Monitors CPU/memory usage

### 📝 Output

- **Logs**: `logs/parallel-research-YYYYMMDD-HHMMSS.log`
- **Analysis Files**: Created in temporary directories, then integrated
- **Git Branches**: Named with timestamp for tracking
- **Reports**: Comprehensive markdown reports generated

### ⚡ Performance

- **Real-time Dashboard**: Updates every 10 seconds during parallel research
- **Progress Tracking**: Shows completion percentage and line counts
- **System Monitoring**: CPU usage and memory consumption
- **Agent Status**: Live status for each research agent

### 🛠 Troubleshooting

```bash
./research status    # Check what's running
./research clean     # Clean up if stuck
./research logs      # View error logs
```

Most issues resolve by cleaning up and restarting.