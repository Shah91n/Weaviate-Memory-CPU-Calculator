# Weaviate Memory & CPU Calculator <img alt='Weaviate logo' src='https://weaviate.io/img/site/weaviate-logo-light.png' width='148' align='right' />

[![Weaviate](https://img.shields.io/static/v1?label=for&message=Weaviate%20%E2%9D%A4&color=green&style=flat-square)](https://weaviate.io/)
[![GitHub Repo stars](https://img.shields.io/github/stars/Shah91n/Weaviate-Memory-CPU-Calculator?style=social)](https://github.com/Shah91n/Weaviate-Memory-CPU-Calculator)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://weaviate-memory-cpu-calculator.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)

**🎯 Memory & CPU planning tool for Weaviate vector database deployments**

Beginner-friendly Estimator based on [official Weaviate documentation](https://weaviate.io/developers/weaviate/concepts/resources) to help you plan memory and CPU requirements for your vector database deployment.

## ✨ Key Features

### 🧮 **Accurate Calculations**
- **Memory**: 2x GC overhead rule + HNSW connections (1.5x avg)
- **CPU**: Official formula `1000ms ÷ latency × 80% efficiency`
- **Storage**: Vector + metadata + 20% overhead
- **All compression types**: PQ, BQ, SQ, RQ 8-bit, RQ 1-bit

### 🎯 **Smart Planning**
- **Performance-based**: Uses target QPS + expected latency
- **Compression comparison**: Side-by-side memory savings
- **Deployment recommendations**: Docker → Kubernetes based on scale
- **Optimization tips**: Automatic suggestions for your config

### 🗜️ **Compression Support**
| Method | Memory Saved | Training | Best For |
|--------|--------------|----------|----------|
| **PQ** | 85% | ✅ | Best balance |
| **BQ** | 97% | ❌ | Maximum savings |
| **SQ** | 75% | ✅ | Fast compression |
| **RQ 8-bit** | 75% | ❌ | No training, 8-bit |
| **RQ 1-bit** | 97% | ❌ | No training, max savings |

## 🚀 Quick Start

### Option 1: Web App (Recommended)
**[Visit the live calculator](https://weaviate-memory-cpu-calculator.streamlit.app/)** - No installation needed!

### Option 2: Local Installation (For custom modifications or advanced use)

```bash
# Clone and setup
git clone https://github.com/Shah91n/Weaviate-Memory-CPU-Calculator.git
cd Weaviate-Memory-CPU-Calculator
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
# Open: http://localhost:8501
```

### Option 3: Docker

```bash
# Build and run
docker build -t weaviate-calculator .
docker run -p 8501:8501 weaviate-calculator
```

## 📊 How It Works

### Memory Calculation
```python
# Vector memory (with 2x GC overhead)
vector_memory = objects × dimensions × 4 bytes × 2

# HNSW connections (1.5x average)
connections = objects × maxConnections × 1.5 × 10 bytes

# Total memory
total = vector_memory + connections
```

### CPU Calculation
```python
# Official Weaviate formula
qps_per_core = 1000ms ÷ expected_latency_ms × 0.8
min_cores = target_qps ÷ qps_per_core
recommended = min_cores × 2  # Headroom for imports
```

## 🔗 Related Tools

- **[Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/)** - Detailed disk planning
- **[Weaviate Disk Calculator Source](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)** - Open source

## 📚 Official References

- **[Resource Planning](https://weaviate.io/developers/weaviate/concepts/resources)** - Core formulas
- **[Compression Guide](https://weaviate.io/developers/weaviate/starter-guides/managing-resources/compression)** - All methods
- **[Vector Quantization](https://weaviate.io/developers/weaviate/concepts/vector-quantization)** - Technical details

## 🤝 Contributing

Found a bug or want to improve something? Contributions welcome!

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

**💡 Made for the Weaviate Users & community** | Star ⭐ if this helped you!
