import time
import statistics

from rag import generate_answer


questions = [
    "What is Voice RAG?",
    "What is RAG?",
    "How does the system work?",
    "What does Voice RAG allow users to do?",
    "How does the system generate answers?"
]


latencies = []

print("\n==============================")
print("VOICE RAG PERFORMANCE TEST")
print("==============================\n")


for question in questions:

    start = time.perf_counter()

    generate_answer(question)

    end = time.perf_counter()

    latency = (end - start) * 1000

    latencies.append(latency)

    print(f"Question: {question}")
    print(f"Latency: {latency:.2f} ms")
    print("------------------------------")


latencies.sort()


def percentile(data, percentile):

    index = int((percentile / 100) * len(data))

    index = min(index, len(data) - 1)

    return data[index]


print("\n==============================")
print("RESULTS")
print("==============================")

print(f"P50  : {percentile(latencies, 50):.2f} ms")
print(f"P70  : {percentile(latencies, 70):.2f} ms")
print(f"P100 : {max(latencies):.2f} ms")

print(f"\nAverage: {statistics.mean(latencies):.2f} ms")