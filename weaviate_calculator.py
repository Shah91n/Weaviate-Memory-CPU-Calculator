from dataclasses import dataclass
from enum import Enum
import math

class CompressionType(Enum):
	NONE = "none"
	PQ = "pq" # Product Quantization: 85% reduction, requires training
	BQ = "bq" # Binary Quantization: 97% reduction, no training
	SQ = "sq" # Scalar Quantization: 75% reduction, requires training  
	RQ_8BIT = "rq_8bit" # Rotational Quantization 8-bit: 75% reduction, no training (v1.32+)
	RQ_1BIT = "rq_1bit" # Rotational Quantization 1-bit: 97% reduction, no training (v1.32+)

@dataclass
class ResourceEstimate:
	memory_gb: float
	memory_gb_with_pq: float
	memory_gb_with_bq: float
	memory_gb_with_sq: float
	memory_gb_with_rq_8bit: float
	memory_gb_with_rq_1bit: float
	disk_storage_gb: float
	min_cpu_cores: int
	recommended_cpu_cores: int
	go_heap_gb: float
	gomemlimit_gb: float
	vectors_memory_gb: float
	connections_memory_gb: float
	vector_data_gb: float
	vector_overhead_gb: float

class WeaviateResourceCalculator:
	"""
	Calculate Weaviate resource requirements based on official documentation.
	
	Uses formulas from: https://weaviate.io/developers/weaviate/concepts/resources
	"""

	def __init__(self):
		self.bytes_per_float32 = 4
		self.bytes_per_connection = 4  # 2-5B variable encoding per HNSW connection
		self.vector_overhead_bytes = 30  # Per-vector cache overhead
		self.heap_buffer_gb = 2.0
		self.gomemlimit_headroom = 1.2  # 20% headroom over Go Heap
		self.container_gomemlimit_ratio = 0.8  # GOMEMLIMIT sits at 80% of container

	def calculate_vector_memory(self, num_objects: int, dimensions: int) -> float:
		"""
		Vector cache memory: objects × (dimensions × 4 + 30) bytes.
		The 30-byte overhead per vector covers cache metadata.
		"""
		total_bytes = num_objects * (dimensions * self.bytes_per_float32 + self.vector_overhead_bytes)
		return total_bytes / (1024 ** 3)

	def calculate_hnsw_connections_memory(self, num_objects: int, max_connections: int) -> float:
		"""
		HNSW graph memory: objects × avg_connections × bytes_per_connection.
		Base layer uses 2×maxConnections, upper layers 1×, averaging ~1.5×.
		Connection encoding uses 2-5 bytes (variable based on index size).
		"""
		avg_connections_per_object = max_connections * 1.5
		total_bytes = num_objects * avg_connections_per_object * self.bytes_per_connection
		return total_bytes / (1024 ** 3)

	def calculate_total_memory(self, vector_data_gb: float, vector_overhead_gb: float,
		connections_memory_gb: float,
		compression: CompressionType = CompressionType.NONE) -> tuple:
		"""
		Go Heap → GOMEMLIMIT → Container memory.
		Compression reduces the vector data portion; per-vector overhead stays.
		Go Heap   = compressed vectors + connections + 2 GB buffer
		GOMEMLIMIT = Go Heap × 1.2
		Container  = GOMEMLIMIT / 0.8
		"""
		compression_factors = {
			CompressionType.NONE: 1.0,
			CompressionType.PQ: 0.15,
			CompressionType.BQ: 0.03,
			CompressionType.SQ: 0.25,
			CompressionType.RQ_8BIT: 0.25,
			CompressionType.RQ_1BIT: 0.03,
		}
		factor = compression_factors.get(compression, 1.0)
		compressed_vectors = vector_data_gb * factor + vector_overhead_gb
		go_heap = compressed_vectors + connections_memory_gb + self.heap_buffer_gb
		gomemlimit = go_heap * self.gomemlimit_headroom
		container = gomemlimit / self.container_gomemlimit_ratio
		return go_heap, gomemlimit, container

	def calculate_disk_storage(self, num_objects: int, dimensions: int, 
		object_size_kb: float = 4.0, compression: CompressionType = CompressionType.NONE) -> float:
		"""
		Disk storage: with compression both original and compressed vectors are stored
		(rescoring reads originals for final accuracy).
		"""
		vector_bytes = num_objects * dimensions * self.bytes_per_float32
		vector_gb = vector_bytes / (1024 ** 3)

		if compression != CompressionType.NONE:
			compression_factors = {
				CompressionType.PQ: 0.15,
				CompressionType.BQ: 0.03,
				CompressionType.SQ: 0.25,
				CompressionType.RQ_8BIT: 0.25,
				CompressionType.RQ_1BIT: 0.03,
			}
			compressed_factor = compression_factors.get(compression, 1.0)
			total_vector_gb = vector_gb + (vector_gb * compressed_factor)
		else:
			total_vector_gb = vector_gb

		metadata_bytes = num_objects * object_size_kb * 1024
		metadata_gb = metadata_bytes / (1024 ** 3)

		raw_storage = total_vector_gb + metadata_gb
		return raw_storage * 1.2

	def calculate_cpu_requirements(self, target_qps: int, expected_latency_ms: int) -> tuple:
		"""
		QPS/core = 1000ms / latency × 0.8 efficiency.
		Recommended = min_cores × 2 for import headroom and peaks.
		"""
		theoretical_qps_per_core = 1000.0 / expected_latency_ms
		realistic_qps_per_core = theoretical_qps_per_core * 0.8
		min_cores = max(1, math.ceil(target_qps / realistic_qps_per_core))
		recommended_cores = min_cores * 2
		return min_cores, recommended_cores

	def get_deployment_recommendation(self, num_objects: int, memory_gb: float) -> tuple:
		if num_objects <= 100_000:
			return (
				"Single Docker Container", 
				"4-8 GB RAM, 2-4 CPU cores",
				"Perfect for development and small projects"
			)
		elif num_objects <= 1_000_000:
			return (
				"Docker Compose", 
				"8-16 GB RAM, 4-8 CPU cores", 
				"Good for production with <1M vectors"
			)
		elif num_objects <= 10_000_000:
			return (
				"Kubernetes (3 nodes)", 
				"32-64 GB RAM, 16-32 CPU cores per node",
				"Recommended for serious production workloads"
			)
		else:
			return (
				"Kubernetes Cluster (3+ nodes)", 
				"128+ GB RAM, 32+ CPU cores per node",
				"Enterprise-scale deployment with high availability"
			)

	def get_optimization_tips(self, num_objects: int, dimensions: int, max_connections: int, 
		memory_gb: float, compression: CompressionType) -> list:
		tips = []

		if memory_gb > 50 and compression == CompressionType.NONE:
			tips.append("💡 Consider enabling Product Quantization (PQ) to reduce vector memory by ~85%")

		if memory_gb > 100 and compression in [CompressionType.NONE, CompressionType.SQ, CompressionType.RQ_8BIT, CompressionType.RQ_1BIT]:
			tips.append("💡 For extreme savings, try BQ or RQ 1-bit for ~97% vector memory reduction")

		if max_connections > 32 and dimensions >= 768:
			tips.append("💡 Reduce maxConnections to 16-32 for high-dimensional vectors (768D+)")

		if dimensions > 1536:
			tips.append("💡 Consider a lower-dimensional embedding model if accuracy allows")

		if num_objects > 1_000_000:
			tips.append("💡 Use multiple shards for better import performance on large datasets")

		if num_objects > 10_000_000:
			tips.append("💡 Consider horizontal scaling with multiple Weaviate nodes")

		tips.append("💡 Always set GOMEMLIMIT to 80% of container memory to prevent OOM kills")
		tips.append("💡 Set vectorCacheMaxObjects to control cache growth instead of relying on defaults")
		tips.append("💡 Monitor go_memstats_heap_inuse_bytes — scale when it exceeds 80% of GOMEMLIMIT")

		return tips

	def get_recommended_resources(self, num_objects: int, dimensions: int,
		target_qps: int = 50, expected_latency_ms: int = 50,
		max_connections: int = 32, 
		compression: CompressionType = CompressionType.NONE,
		object_size_kb: float = 4.0) -> ResourceEstimate:
		vector_data_gb = (num_objects * dimensions * self.bytes_per_float32) / (1024 ** 3)
		vector_overhead_gb = (num_objects * self.vector_overhead_bytes) / (1024 ** 3)
		vectors_memory_gb = vector_data_gb + vector_overhead_gb
		connections_memory_gb = self.calculate_hnsw_connections_memory(num_objects, max_connections)

		go_heap_gb, gomemlimit_gb, memory_no_compression = self.calculate_total_memory(
			vector_data_gb, vector_overhead_gb, connections_memory_gb, CompressionType.NONE)
		_, _, memory_with_pq = self.calculate_total_memory(
			vector_data_gb, vector_overhead_gb, connections_memory_gb, CompressionType.PQ)
		_, _, memory_with_bq = self.calculate_total_memory(
			vector_data_gb, vector_overhead_gb, connections_memory_gb, CompressionType.BQ)
		_, _, memory_with_sq = self.calculate_total_memory(
			vector_data_gb, vector_overhead_gb, connections_memory_gb, CompressionType.SQ)
		_, _, memory_with_rq_8bit = self.calculate_total_memory(
			vector_data_gb, vector_overhead_gb, connections_memory_gb, CompressionType.RQ_8BIT)
		_, _, memory_with_rq_1bit = self.calculate_total_memory(
			vector_data_gb, vector_overhead_gb, connections_memory_gb, CompressionType.RQ_1BIT)

		disk_storage_gb = self.calculate_disk_storage(num_objects, dimensions, object_size_kb, compression)
		min_cpu_cores, recommended_cpu_cores = self.calculate_cpu_requirements(target_qps, expected_latency_ms)

		return ResourceEstimate(
			memory_gb=memory_no_compression,
			memory_gb_with_pq=memory_with_pq,
			memory_gb_with_bq=memory_with_bq,
			memory_gb_with_sq=memory_with_sq,
			memory_gb_with_rq_8bit=memory_with_rq_8bit,
			memory_gb_with_rq_1bit=memory_with_rq_1bit,
			disk_storage_gb=disk_storage_gb,
			min_cpu_cores=min_cpu_cores,
			recommended_cpu_cores=recommended_cpu_cores,
			go_heap_gb=go_heap_gb,
			gomemlimit_gb=gomemlimit_gb,
			vectors_memory_gb=vectors_memory_gb,
			connections_memory_gb=connections_memory_gb,
			vector_data_gb=vector_data_gb,
			vector_overhead_gb=vector_overhead_gb,
		)

