# Save as test_examples.py
from src.data.data_preprocessor import TextPreprocessor
from src.data.feature_engineer import CombinedFeatureExtractor
import joblib

# Load models
classifier = joblib.load('models/classifier.pkl')
scaler = joblib.load('models/scaler.pkl')
extractor = joblib.load('models/feature_extractor.pkl')

# Test different problem descriptions
test_problems = [
    ("Easy", "Given an array of integers, find the sum of all elements."),
    ("Easy", "Print 'Hello World' to the console."),
    ("Easy", "Check if a number is even or odd."),
    ("Medium", "Given the root of a binary tree, return the level order traversal of its nodes' values."),
    ("Medium", "You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees."),
    ("Medium", "Group the anagrams together using a hash map."),

    # New Hard Candidates
    # --- GRAPH THEORY & NETWORK FLOW ---
    ("Hard", "Find the maximum flow through a flow network from source to sink using the Ford-Fulkerson algorithm with BFS."),
    ("Hard", "There are N cities connected by M flights. Each flight has a price. Find the cheapest price from src to dst with at most K stops using Bellman-Ford or Dijkstra."),
    ("Hard", "Given a list of airline tickets, reconstruct the itinerary in lexical order using Hierholzer's algorithm for Eulerian paths."),
    ("Hard", "Find the shortest path visiting all nodes in a weighted graph using bitmask dynamic programming."),

    # --- ADVANCED DYNAMIC PROGRAMMING ---
    ("Hard", "Given n balloons, indexed from 0 to n-1. Each balloon is painted with a number. Burst all the balloons to maximize coins obtained. Solve using matrix chain multiplication DP."),
    ("Hard", "Determine if a string s matches a pattern p containing support for '.' and '*' where '*' matches zero or more of the preceding element."),
    ("Hard", "Given a string s, partition s such that every substring of the partition is a palindrome. Return the minimum cuts needed for a palindrome partitioning."),
    ("Hard", "Maximize the total profit by buying and selling stocks with at most k transactions allowed."),

    # --- TREE & TRIE & DATA STRUCTURES ---
    ("Hard", "Design a data structure that supports adding new words and finding if a string matches any previously added string using a Trie (Prefix Tree)."),
    ("Hard", "Serialize a binary tree to a string and deserialize it back to a tree structure using preorder traversal."),
    ("Hard", "Given an integer array nums, return the number of reverse pairs in the array. A reverse pair is (i, j) where nums[i] > 2 * nums[j]. Use Merge Sort or Fenwick Tree."),
    ("Hard", "Given an n x m board of characters and a list of strings, return all words on the board using DFS and a Trie."),

    # --- GEOMETRY & MATH ---
    ("Hard", "Given a set of points in a 2D plane, find the convex hull of the set of points using the Monotone Chain algorithm."),
    ("Hard", "Find the median of two sorted arrays of different sizes with O(log(m+n)) runtime complexity."),

    # --- HARD STACK/SLIDING WINDOW ---
    ("Hard", "Given an array of integers heights representing the histogram's bar height, return the area of the largest rectangle in the histogram using a monotonic stack."),
    ("Hard", "Find the maximum element in every sliding window of size k moving from left to right across an array.")
]

preprocessor = TextPreprocessor()

for expected, desc in test_problems:
    text = preprocessor.combine_fields("", desc, "", "")
    features = extractor.transform([text])
    scaled = scaler.transform(features)
    
    pred_class = classifier.predict(scaled)[0]
    proba = classifier.predict_proba(scaled)[0]
    
    match = "✓" if str(pred_class) == expected.lower() else "✗"
    print(f"{match} Expected: {expected:6} | Got: {str(pred_class):6} | Conf: {proba.max():.1%} | {desc[:50]}")