import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import os
from weaviate_calculator import (
	WeaviateResourceCalculator, 
	CompressionType, 
	EMBEDDING_MODELS,
	ResourceEstimate
)

logo_path = os.path.join("assets", "weaviate-logo.png")
logo_image = Image.open(logo_path)

# Page configuration
st.set_page_config(
	page_title="Weaviate Resource Calculator",
	page_icon=logo_image,
	layout="wide",
	initial_sidebar_state="expanded"
)

# Initialize session state
if 'calculator' not in st.session_state:
	st.session_state.calculator = WeaviateResourceCalculator()

if 'results' not in st.session_state:
	st.session_state.results = None

def clear_session_state():
	"""Clear all session state and reset the app"""
	print("clear_session_state called")
	for key in list(st.session_state.keys()):
		del st.session_state[key]
	st.cache_data.clear()
	st.rerun()

def format_number(num: float, decimals: int = 2) -> str:
	"""Format number with appropriate units"""
	if num >= 1e9:
		return f"{num/1e9:.{decimals}f}B"
	elif num >= 1e6:
		return f"{num/1e6:.{decimals}f}M"
	elif num >= 1e3:
		return f"{num/1e3:.{decimals}f}K"
	else:
		return f"{num:.{decimals}f}"

def main():

	st.markdown(
	    "<h1 style='text-align: center;'>Weaviate Memory & CPU Resource Estimator</h1>",
	    unsafe_allow_html=True
	)
	st.markdown("""
		        **Resource planning tool** based on [official Weaviate documentation](https://weaviate.io/developers/weaviate/concepts/resources)
		        """)

	# Center the buttons by using empty columns on the sides
	col_left, col_center1, col_center2, col_right = st.columns([2, 1, 1, 2])

	with col_center1:
		calculate_clicked = st.button("🔄 Calculate Resources", use_container_width=True, type="primary")

	with col_center2:
		clear_clicked = st.button("🗑️ Clear All", help="Reset all inputs and calculations", use_container_width=True)

	# Handle clear button
	if clear_clicked:
		clear_session_state()

	# Create tabs
	tab1, tab2, tab3 = st.tabs(["📟 Estimator", "📖 How It Works", "📚 References"])

	with tab1:
		calculator_tab(calculate_clicked)

	with tab2:
		how_it_works_tab()

	with tab3:
		references_tab()

def calculator_tab(calculate_clicked=False):
	"""Main calculator interface"""
	col1, col2 = st.columns([1, 2])

	with col1:
		st.subheader("📥 Please fill in the parameters...")

		# Basic parameters
		num_objects = st.number_input(
			"Number of vectors (number of objects * vectors per object)",
			min_value=10000,
			max_value=100_000_000_000,
			value=50_000_000,
			step=100_000,
			help="Total Number of vectors (number of objects * vectors per object) to store in Weaviate. E.g. 10000 objects with 1536 dimensions each is 15360000 vectors."
		)

		# Vector dimensions
		st.markdown("**Vector Dimensions**")
		dimension_source = st.radio(
			"Select dimension source:",
			["Embedding Model", "Custom Dimensions"],
			horizontal=True
		)

		if dimension_source == "Embedding Model":
			provider = st.selectbox("Provider", list(EMBEDDING_MODELS.keys()))
			model = st.selectbox("Model", list(EMBEDDING_MODELS[provider].keys()))
			dimensions = EMBEDDING_MODELS[provider][model]
			st.info(f"**{dimensions} dimensions** for {model}")
		else:
			dimensions = st.number_input(
				"Enter Vector Dimensions",
				min_value=1,
				max_value=10000,
				value=768,
				step=1,
				help="Enter the number of dimensions per vector (e.g., 768, 1536, 3072)"
			)
			st.info(f"**{dimensions} dimensionality**")

		st.markdown("**Performance Requirements**")
		target_qps = st.number_input(
			"Target Queries Per Second (QPS)",
			min_value=1,
			max_value=10000,
			value=50,
			help="Expected query throughput"
		)

		expected_latency = st.slider(
			"Expected Query Latency (ms)",
			min_value=10,
			max_value=1000,
			value=50,
			step=10,
			help="Expected time per query in milliseconds"
		)
		st.caption("Lower latency increases QPS per core (1000 ÷ latency). For the same target QPS: lower latency → fewer cores; higher latency → more cores.")

		with st.expander("⚙️ Advanced Settings", expanded=False):
			st.markdown("**HNSW Index Configuration**")
			max_connections = st.slider(
				"maxConnections",
				min_value=4,
				max_value=128,
				value=32,
				step=4,
				help="HNSW graph connections per node. Lower values reduce memory but may impact recall."
			)

		object_size_kb = st.number_input(
			"Average Object Size (KB)",
			min_value=0.1,
			max_value=100.0,
			value=4.0,
			step=0.5,
			help="Average size of non-vector data per object"
		)

	# Handle calculation when button is clicked
	if calculate_clicked:
		results = st.session_state.calculator.get_recommended_resources(
			num_objects=int(num_objects),
			dimensions=dimensions,
			target_qps=target_qps,
			expected_latency_ms=expected_latency,
			max_connections=max_connections,
			compression=CompressionType.NONE,  # Default for disk calculation
			object_size_kb=object_size_kb
		)
		# Store results in session state
		st.session_state.results = results
		st.session_state.current_params = {
			'num_objects': num_objects,
			'dimensions': dimensions,
			'max_connections': max_connections,
			'target_qps': target_qps,
			'expected_latency': expected_latency,
			'object_size_kb': object_size_kb
		}
		# Force rerun to show results immediately
		st.rerun()

	with col2:
		if st.session_state.results:
			display_results(st.session_state.results, st.session_state.current_params)
		else:
			st.info("👈 Configure parameters and click 'Calculate Resources' to see results")

def display_results(results: ResourceEstimate, params: dict):
	"""Display calculation results"""
	st.subheader("📊 Resource Requirements")
	
	st.info("⚠️ **Important:** This estimate is for the HNSW index only. The flat index will use dramatically less RAM as it performs brute-force searches from disk.")

	st.subheader("📊 Memory Requirements")
	
	col1, col2, col3 = st.columns(3)

	# Key metrics - All compression types
	with col1:
		st.metric(
			"No Compression",
			f"{results.memory_gb:.1f} GB",
			help="Full precision vectors"
		)
		st.metric(
			"Product Quantization (PQ)",
			f"{results.memory_gb_with_pq:.1f} GB",
			f"-{((1 - results.memory_gb_with_pq/results.memory_gb) * 100):.0f}%",
			help="85% reduction, requires training"
		)

	with col2:
		st.metric(
			"Binary Quantization (BQ)",
			f"{results.memory_gb_with_bq:.1f} GB",
			f"-{((1 - results.memory_gb_with_bq/results.memory_gb) * 100):.0f}%",
			help="97% reduction, no training"
		)
		st.metric(
			"Scalar Quantization (SQ)",
			f"{results.memory_gb_with_sq:.1f} GB",
			f"-{((1 - results.memory_gb_with_sq/results.memory_gb) * 100):.0f}%",
			help="75% reduction, requires training"
		)

	with col3:
		st.metric(
			"RQ 8-bit",
			f"{results.memory_gb_with_rq_8bit:.1f} GB",
			f"-{((1 - results.memory_gb_with_rq_8bit/results.memory_gb) * 100):.0f}%",
			help="75% reduction, no training"
		)
		st.metric(
			"RQ 1-bit",
			f"{results.memory_gb_with_rq_1bit:.1f} GB",
			f"-{((1 - results.memory_gb_with_rq_1bit/results.memory_gb) * 100):.0f}%",
			help="97% reduction, no training"
		)

	# System requirements
	st.subheader("💻 CPU & Disk Requirements")
	col1, col2, col3 = st.columns(3)

	with col1:
		st.metric(
			"Disk Storage",
			f"{results.disk_storage_gb:.1f} GB",
			help="Including 20% overhead for indexes"
		)
		
		st.info("**For detailed disk calculations, visit 🔗 :** [Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/)")

	with col2:
		st.metric(
			"Min CPU Cores",
			f"{results.min_cpu_cores}",
			help="For target QPS with efficiency factor"
		)

	with col3:
		st.metric(
			"Recommended CPU Cores",
			f"{results.recommended_cpu_cores}",
			help="With headroom for imports and peaks"
		)

	# Compression comparison
	st.markdown("---")
	st.subheader("🗜️ Compression Options Comparison")

	compression_data = {
		"Method": ["No Compression", "Product Quantization (PQ)", "Binary Quantization (BQ)", "Scalar Quantization (SQ)", "Rotational Quantization 8-bit (RQ)", "Rotational Quantization 1-bit (RQ)"],
		"Memory (GB)": [results.memory_gb, results.memory_gb_with_pq, results.memory_gb_with_bq, results.memory_gb_with_sq, results.memory_gb_with_rq_8bit, results.memory_gb_with_rq_1bit],
		"Reduction": ["0%", "85%", "97%", "75%", "75%", "97%"],
		"Training Required": ["❌", "✅", "❌", "✅", "❌", "❌"],
		"Notes": ["Full precision", "Best balance", "Maximum savings", "Fast compression", "No training, 8-bit", "No training, 1-bit"]
	}

	compression_df = pd.DataFrame(compression_data)
	st.dataframe(compression_df, use_container_width=True)

	# Memory breakdown
	st.markdown("---")
	st.subheader("💾 Memory Breakdown")

	col1, col2 = st.columns(2)

	with col1:
		# Memory composition chart
		fig = go.Figure(data=[
			go.Bar(
				name='Vectors',
				x=['No Compression', 'With PQ'],
				y=[results.vectors_memory_gb * 2, results.vectors_memory_gb * 0.15 * 2],
				text=[f"{results.vectors_memory_gb * 2:.1f} GB", 
					f"{results.vectors_memory_gb * 0.15 * 2:.1f} GB"],
				textposition='auto',
			),
			go.Bar(
				name='HNSW Connections',
				x=['No Compression', 'With PQ'],
				y=[results.connections_memory_gb, results.connections_memory_gb],
				text=[f"{results.connections_memory_gb:.1f} GB", 
					f"{results.connections_memory_gb:.1f} GB"],
				textposition='auto',
			)
		])

		fig.update_layout(
			title="Memory Composition",
			yaxis_title="Memory (GB)",
			barmode='stack',
			height=400
		)
		st.plotly_chart(fig, use_container_width=True)

	with col2:
		# Calculation details
		st.markdown("**📐 Calculation Details**")

		st.code(f"""
		             # Vector Memory (No Compression)
		             Dimensions: {params['dimensions']}
		             Objects: {format_number(params['num_objects'])}
		             Bytes per vector: {params['dimensions']} × 4 = {params['dimensions'] * 4:,} bytes
		             Total vector memory: {results.vectors_memory_gb:.2f} GB
		             With GC overhead (2x): {results.vectors_memory_gb * 2:.2f} GB
		         
		             # HNSW Connections Memory
		             Max connections: {params['max_connections']}
		             Avg connections: {params['max_connections'] * 1.5:.0f} connections
		             Bytes per connection: 10
		             Total connections memory: {results.connections_memory_gb:.2f} GB
		         
		             # Total Memory
		             Rule of thumb: {results.memory_gb:.2f} GB
		             Recommended (+25%): {results.recommended_memory_gb:.2f} GB
		         """, language="python")

		st.markdown("**⚡ CPU Calculation**")
		st.info("Latency ↔ CPU cores: QPS/core ≈ 1000 ÷ latency. So at 10ms ≈ 100 QPS/core → fewer cores; at 1000ms ≈ 1 QPS/core → more cores, for the same target QPS.")

		target_qps = params['target_qps']
		expected_latency = params['expected_latency']

		theoretical_qps_per_core = 1000.0 / expected_latency
		realistic_qps_per_core = theoretical_qps_per_core * 0.8

		st.code(f"""
		             # CPU Requirements
		             Target QPS: {target_qps}
		             Expected latency: {expected_latency}ms
		             Theoretical QPS/core: 1000ms ÷ {expected_latency}ms = {theoretical_qps_per_core:.1f}
		             Real-world QPS/core: {theoretical_qps_per_core:.1f} × 0.8 = {realistic_qps_per_core:.1f}
		             Min cores needed: {target_qps} ÷ {realistic_qps_per_core:.1f} = {results.min_cpu_cores}
		             Recommended: {results.min_cpu_cores} × 2 = {results.recommended_cpu_cores} cores
		         """, language="python")

		st.markdown("**💿 Disk Storage Calculation**")

		st.code(f"""
		             # Basic Disk Storage
		             Vector storage: {results.vectors_memory_gb:.2f} GB
		             Objects: {format_number(params['num_objects'])} × 4KB = {(params['num_objects'] * 4 / 1024 / 1024):.2f} GB
		             System overhead (20%): +{results.disk_storage_gb * 0.2:.2f} GB
		             Total disk: {results.disk_storage_gb:.2f} GB
		         
		             Note: With compression, both original + compressed stored
		         """, language="python")

		st.info("🔗 **Need advanced or custom disk storage estimates?** Try 🔗 [Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/) for in-depth scenarios and edge cases. [View Source Code](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)")

	# Recommendations
	st.markdown("---")
	st.subheader("🎯 Deployment Recommendations")

	deployment_type, instance_rec, notes = st.session_state.calculator.get_deployment_recommendation(
		params['num_objects'], results.memory_gb
	)

	col1, col2 = st.columns(2)
	with col1:
		st.info(f"**Recommended Deployment:** {deployment_type}")
	with col2:
		st.info(f"**Instance Sizing:** {instance_rec}")

	# Cost optimization tips
	optimization_tips = st.session_state.calculator.get_optimization_tips(
		params['num_objects'], params['dimensions'], params['max_connections'],
		results.memory_gb, CompressionType.NONE
	)

	if optimization_tips:
		st.markdown("### 💡 Optimization Tips")
		for tip in optimization_tips:
			st.markdown(tip)

def how_it_works_tab():
	"""Combined how it works and guide"""
	st.markdown("""
	            ### 🎯 Quick Planning Guide
	            
	            **New to vector databases?** Here's what you need to know:
	            
	            - **Vectors** = Lists of numbers representing your data's meaning
	            - **Dimensions** = How many numbers in each vector (more = better accuracy, more memory)
	            - **HNSW** = The search algorithm that connects similar vectors for fast searching
	            - **Compression** = Reduces memory usage but may reduce accuracy slightly
	            """)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("""
		            #### 🔢 Understanding Dimensions
		            
		            | Dimensions | Use Case | Memory Impact |
		            |------------|----------|---------------|
		            | 384 | Small projects | Low |
		            | 768 | General purpose | Medium |
		            | 1536 | High accuracy | High |
		            | 3072 | Maximum quality | Very High |
		            
		            **Rule:** More dimensions = better accuracy but more memory needed
		            """)

	with col2:
		st.markdown("""
	            #### 🗜️ Compression Guide
	            
	            | Method | Memory Saved | When to Use |
	            |--------|--------------|-------------|
	            | PQ | 85% | Best balance |
	            | SQ | 75% | Fast setup |
	            | BQ | 97% | Maximum savings |
	            | RQ 8-bit | 75% | No training |
	            | RQ 1-bit | 97% | No training, max savings |
	            
	            **Rule:** Always consider compression for >5M vectors
	            """)

	st.markdown("""
	            ---
	            ### 🧮 How Calculations Work
	                
	            #### Memory Rule of Thumb
	            ```
	            Memory = 2 × (Number of vectors × Dimensions × 4 bytes) + HNSW connections
	            ```
	            
	            **The 2x multiplier accounts for:**
	            - Go's garbage collection overhead during imports
	            - Temporary memory allocations
	            - Safety buffer for production stability
	            
	            #### Detailed Memory Formula
	            ```python
	            # Vector memory
	            vector_memory = objects × dimensions × 4 bytes
	            
	            # HNSW connections memory
	            # Base layer: 2 × maxConnections, Upper layers: 1 × maxConnections
	            # Average: ~1.5 × maxConnections
	            connections_memory = objects × (maxConnections × 1.5) × 10 bytes
	            
	            # Total memory with GC overhead
	            total_memory = (vector_memory × 2) + connections_memory
	            ```
	            
	            #### CPU Requirements
	            ```python
	            # Official Weaviate formula from benchmark documentation
	            theoretical_qps_per_core = 1000ms ÷ query_latency_ms
	            
	            # Apply real-world efficiency (80% due to synchronization overhead)
	            realistic_qps_per_core = theoretical_qps_per_core × 0.8
	            
	            # Calculate cores needed
	            min_cores = target_qps ÷ realistic_qps_per_core
	            recommended_cores = min_cores × 2  # headroom for imports and peaks
	            ```
	            
	            **Key facts from Weaviate docs:**
	            - Each search is single-threaded, but multiple searches use multiple threads
	            - "When search throughput is limited, add CPUs to increase QPS"
	            - Import operations are also CPU-bound (building HNSW index)
	            - Real-world efficiency is ~80% due to synchronization mechanisms
	            
	            #### Storage Calculation
	            ```python
	            # With compression, both original and compressed vectors stored
	            vector_storage = objects × dimensions × 4 bytes
	            if compression_enabled:
	                total_vectors = original_vectors + compressed_vectors
	            
	            total_storage = (vectors + objects) × 1.2  # 20% overhead
	            ```
	            """)
	st.info("🔗 **For detailed disk calculations:** [Weaviate Disk Storage Calculator](https://weaviate-disk-calculator.streamlit.app/) | [Source](https://github.com/Shah91n/Weaviate-Disk-Storage-Calculator)")

# Compression visualization
	st.markdown("---")
	st.subheader("🗜️ Simple example on how compression works")
	
	col1, col2 = st.columns(2)
	
	with col1:
		st.markdown("""
			#### Product Quantization (PQ) Process
			
			```
			Original Vector (768 dimensions)
			├── 3072 bytes (768 × 4 bytes)
			│
			├── Split into 128 segments
			│   Each segment: 6 dimensions
			│
			├── Find closest centroid (256 options)
			│   Store centroid ID (1 byte)
			│
			└── Compressed: 128 bytes (128 × 1 byte)
			
			Compression: 3072 → 128 bytes (24x smaller!)
			```
			""", unsafe_allow_html=True)
		
	with col2:
		st.markdown("""
			#### Binary Quantization (BQ)
			
			```
			Original Vector (768 dimensions)
			├── 3072 bytes (768 × 4 bytes)
			│
			├── Convert to binary
			│   Each dimension → 0 or 1
			│
			├── Pack into bits
			│   96 bytes (768 ÷ 8 bits)
			│
			└── Compressed: 96 bytes
			
			Compression: 3072 → 96 bytes (32x smaller!)
			```
			""", unsafe_allow_html=True)
	st.markdown("""
	            ---
	            ### 📋 Step-by-Step Planning
	            
	            1. **Count Your Data**: How many documents/items do you have?
	            2. **Choose Embedding Model**: Pick based on accuracy vs speed needs
	            3. **Calculate Base Memory**: Objects × Dimensions × 4 bytes
	            4. **Add Safety Buffer**: Multiply by 2 for garbage collection
	            5. **Add HNSW Memory**: For the search graph connections
	            6. **Consider Compression**: Rotational Quantization (RQ) offers 4x compression with 98-99% recall
	            7. **Plan Deployment**: Docker for <1M, Kubernetes for >1M objects
	            
	            ### 💡 Best Practices
	            
	            **Memory Optimization:**
	            - Use compression for datasets >5M objects
	            - Reduce maxConnections for high-dimensional vectors (768D+)
	            - Consider lower-dimensional models when possible
	            - RQ (Rotational Quantization) requires no training phase
	            
	            **Performance Tuning:**
	            - Multiple shards improve import speed
	            - Increase efConstruction/ef with lower maxConnections
	            - Use SSDs for storage (required for good performance)
	            - Flat index uses dramatically less RAM than HNSW
	            
	            **Cost Optimization:**
	            - PQ compression can reduce cloud costs by 80%+
	            - RQ provides 4x compression with 98-99% recall
	            - Right-size deployment based on actual usage
	            - Use LIMIT_RESOURCES=true to prevent OOM kills
	            """)

def references_tab():
	"""Updated references with latest documentation"""
	st.markdown("""
	            📚 Documentation References - All calculations in this tool are based on **official Weaviate documentation**:
	            
	            ### Primary Sources
	            
	            1. **[Resource Planning Guide](https://weaviate.io/developers/weaviate/concepts/resources)** - Main memory and CPU formulas
	            2. **[Vector Indexing](https://docs.weaviate.io/weaviate/concepts/indexing)** - HNSW and Flat index details
	            3. **[Compression Guide](https://weaviate.io/developers/weaviate/starter-guides/managing-resources/compression)** - Managing Resources Compression
	            4. **[Vector Quantization](https://weaviate.io/developers/weaviate/concepts/vector-quantization)** - Technical compression details
	            5. **[Rotational Quantization](https://weaviate.io/developers/weaviate/concepts/vector-quantization#rotational-quantization)** - RQ technical details
	            
	            ### Key Formulas Verification
	            
	            #### Memory Rule of Thumb ✅
	            > **"Memory usage = 2 × (the memory footprint of all vectors)"**  
	            
	            #### HNSW Connections ✅  
	            > **"Each object in memory has at most maxConnections connections per layer. Each of the connections uses 8-10B of memory. Note that the base layer allows for 2 * maxConnections."**  
	            
	            ### 🔧 Configuration Examples
	            
	            #### Environment Variables
	            ```bash
	            # Limit Weaviate to 80% of available memory
	            LIMIT_RESOURCES=true
	            
	            # Set Go memory limit (10-20% of total memory for Weaviate)
	            GOMEMLIMIT=2GB
	            
	            # Set maximum CPU threads
	            GOMAXPROCS=16
	            ```
	            
	            #### HNSW Configuration
	            ```json
	            {
	              "vectorIndexConfig": {
	                "maxConnections": 32,        // Reduced for high-dimensional vectors
	                "efConstruction": 128,       // Build-time quality
	                "ef": 100,                   // Query-time quality
	                "dynamicEfMin": 100,
	                "dynamicEfMax": 500,
	                "dynamicEfFactor": 8
	              }
	            }
	            ```
	            
	            ### 🚀 Deployment Guides
	            
	            - [Docker Compose Setup](https://weaviate.io/developers/weaviate/installation/docker-compose)
	            - [Kubernetes Deployment](https://weaviate.io/developers/weaviate/installation/kubernetes)  
	            - [Weaviate Cloud](https://console.weaviate.cloud)
	            - [Environment Variables](https://weaviate.io/developers/weaviate/config-refs/env-vars)
	            
	            ### 📊 Index Types Comparison
	            
	            | Index Type | Memory Usage | Speed | Best For |
	            |------------|--------------|-------|----------|
	            | **HNSW** | High (in-memory) | Very Fast | Large datasets, production |
	            | **Flat** | Low (disk-based) | Slower | Small datasets, testing |
	            
	            **Note:** This calculator estimates for HNSW index. Flat index uses dramatically less RAM.
	            
	            ---
	            
	            > 💡 **Disclaimer:** This calculator is based on formulas and guidance from [Weaviate's official documentation](https://weaviate.io/developers/weaviate/concepts/resources).  
	            The results are intended as practical estimates for most scenarios. For mission-critical or production deployments, always validate with your own data and perform real-world benchmarking.  
  
	            _Tool developed by [Mohamed Shahin](https://github.com/Shah91n)._
	            """)

if __name__ == "__main__":
	main()
