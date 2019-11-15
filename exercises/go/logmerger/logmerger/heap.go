package logmerger

// iteratorsHeap is taken advantaged by the MergedIterator
type iteratorsHeap []LogIterator

func (h iteratorsHeap) Len() int {
	return len(h)
}

// nil is always bigger than any other
func (h iteratorsHeap) Less(i, j int) bool {
	li := h[i].Current()
	lj := h[j].Current()
	if li == nil && lj == nil {
		return false
	} else if li == nil && lj != nil {
		return false
	} else if li != nil && lj == nil {
		return true
	} else {
		return li.commitToken < lj.commitToken
	}
}

func (h iteratorsHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *iteratorsHeap) Push(x interface{}) {
	*h = append(*h, x.(LogIterator))
}

func (h *iteratorsHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// iteratorsHeap is taken advantaged by the MergedIterator
type logsHeap []Log

func (h logsHeap) Len() int {
	return len(h)
}

// nil is always bigger than any other
func (h logsHeap) Less(i, j int) bool {
	return h[i].commitToken < h[j].commitToken
}

func (h logsHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *logsHeap) Push(x interface{}) {
	*h = append(*h, *(x.(*Log)))
}

func (h *logsHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return &x
}
