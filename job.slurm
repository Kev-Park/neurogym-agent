cat > test.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=test_relay
#SBATCH --time=00:05:00
#SBATCH --ntasks=1

curl -s --max-time 5 http://128.112.220.200:7861 && echo reachable || echo unreachable
EOF