import Foundation

/// Minimal binary heap priority queue.
struct PriorityQueue<Element> {
    private var elements: [Element] = []
    private let areSorted: (Element, Element) -> Bool

    init(sort: @escaping (Element, Element) -> Bool) {
        self.areSorted = sort
    }

    var isEmpty: Bool { elements.isEmpty }

    mutating func enqueue(_ value: Element) {
        elements.append(value)
        siftUp(from: elements.count - 1)
    }

    mutating func dequeue() -> Element? {
        guard !elements.isEmpty else { return nil }
        if elements.count == 1 {
            return elements.removeLast()
        }
        let value = elements[0]
        elements[0] = elements.removeLast()
        siftDown(from: 0)
        return value
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = parentIndex(of: child)
        while child > 0 && areSorted(elements[child], elements[parent]) {
            elements.swapAt(child, parent)
            child = parent
            parent = parentIndex(of: child)
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index
        while true {
            let left = leftChildIndex(of: parent)
            let right = rightChildIndex(of: parent)
            var candidate = parent
            if left < elements.count && areSorted(elements[left], elements[candidate]) {
                candidate = left
            }
            if right < elements.count && areSorted(elements[right], elements[candidate]) {
                candidate = right
            }
            if candidate == parent { return }
            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }

    private func parentIndex(of index: Int) -> Int { (index - 1) / 2 }
    private func leftChildIndex(of index: Int) -> Int { (2 * index) + 1 }
    private func rightChildIndex(of index: Int) -> Int { (2 * index) + 2 }
}
