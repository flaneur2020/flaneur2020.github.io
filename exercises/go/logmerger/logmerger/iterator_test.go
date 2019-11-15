package logmerger

import (
	"container/heap"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func makeLogsChan(seqs ...int64) chan Log {
	c1 := make(chan Log, 1024)
	for i := 0; i < len(seqs); i++ {
		c1 <- Log{commitToken: seqs[i], data: []byte(fmt.Sprintf("m%d", seqs[i]))}
	}
	close(c1)
	return c1
}

func convertIteratorToSeqs(it LogIterator) []int64 {
	seqs := []int64{}
	for ; it.Current() != nil; it.Next() {
		seqs = append(seqs, it.Current().commitToken)
	}
	return seqs
}

func Test_ChanIterator(t *testing.T) {
	c1 := makeLogsChan(1, 2, 3)

	it := NewChanIterator(c1)
	assert.Equal(t, it.Current().commitToken, int64(1))
	assert.Equal(t, it.Current().commitToken, int64(1))

	assert.Equal(t, it.Next(), true)
	assert.Equal(t, it.Current().commitToken, int64(2))

	assert.Equal(t, it.Next(), true)
	assert.Equal(t, it.Current().commitToken, int64(3))

	assert.Equal(t, it.Next(), false)
	assert.Equal(t, it.Current(), (*Log)(nil))

	assert.Equal(t, it.Next(), false)
	assert.Equal(t, it.Current(), (*Log)(nil))
}

func Test_IteratorsHeap(t *testing.T) {
	it1 := NewChanIterator(makeLogsChan(1))
	it2 := NewChanIterator(makeLogsChan(2))
	it3 := NewChanIterator(makeLogsChan(3))
	it4 := NewChanIterator(makeLogsChan(4))
	h := iteratorsHeap([]LogIterator{it3, it4, it1, it2})
	heap.Init(&h)

	it := heap.Pop(&h).(LogIterator)
	assert.Equal(t, int64(1), it.Current().commitToken)
	it = heap.Pop(&h).(LogIterator)
	assert.Equal(t, int64(2), it.Current().commitToken)
	it = heap.Pop(&h).(LogIterator)
	assert.Equal(t, int64(3), it.Current().commitToken)
	it = heap.Pop(&h).(LogIterator)
	assert.Equal(t, int64(4), it.Current().commitToken)
}

func Test_MergedIterator(t *testing.T) {
	it1 := NewChanIterator(makeLogsChan(1, 3, 5))
	it2 := NewChanIterator(makeLogsChan(2, 4, 6))
	mt := NewMergedIterator([]LogIterator{it1, it2})

	assert.Equal(t, mt.Next(), true)
	assert.Equal(t, mt.Current().commitToken, int64(1))

	assert.Equal(t, mt.Next(), true)
	assert.Equal(t, mt.Current().commitToken, int64(2))

	assert.Equal(t, mt.Next(), true)
	assert.Equal(t, mt.Current().commitToken, int64(3))

	assert.Equal(t, mt.Next(), true)
	assert.Equal(t, mt.Current().commitToken, int64(4))

	assert.Equal(t, mt.Next(), true)
	assert.Equal(t, mt.Current().commitToken, int64(5))

	assert.Equal(t, mt.Next(), true)
	assert.Equal(t, mt.Current().commitToken, int64(6))

	assert.Equal(t, mt.Next(), false)
	assert.Equal(t, mt.Current(), (*Log)(nil))

	assert.Equal(t, mt.Next(), false)
	assert.Equal(t, mt.Current(), (*Log)(nil))
}

func Test_MergedIterator2(t *testing.T) {
	it1 := NewChanIterator(makeLogsChan(1, 3, 5, 7, 7, 7, 7))
	it2 := NewChanIterator(makeLogsChan(2, 4, 6))
	mt := NewMergedIterator([]LogIterator{it1, it2})
	got := convertIteratorToSeqs(mt)
	want := []int64{1, 2, 3, 4, 5, 6, 7, 7, 7, 7}
	assert.Equal(t, want, got)
}

func Test_MergedIterator3(t *testing.T) {
	it1 := NewChanIterator(makeLogsChan(1, 3, 5, 7, 7, 7, 7))
	it2 := NewChanIterator(makeLogsChan(2, 4, 6))
	it3 := NewChanIterator(makeLogsChan(9, 100))
	it4 := NewChanIterator(makeLogsChan())
	mt := NewMergedIterator([]LogIterator{it1, it2, it3, it4})
	got := convertIteratorToSeqs(mt)
	want := []int64{1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 9, 100}
	assert.Equal(t, want, got)
}

func Test_CommitedLogIterator(t *testing.T) {
	c1 := make(chan Log, 1024)
	c1 <- Log{kind: "prepare", prepareToken: 1, data: []byte("test")}
	c1 <- Log{kind: "commit", prepareToken: 1, commitToken: 2, data: []byte("test")}
	c1 <- Log{kind: "prepare", prepareToken: 3, data: []byte("test")}
	c1 <- Log{kind: "commit", prepareToken: 3, commitToken: 4, data: []byte("test")}
	c1 <- Log{kind: "commit", prepareToken: 3, commitToken: 7, data: []byte("test")}
	close(c1)

	it := NewCommitedLogIterator(NewChanIterator(c1))
	got := convertIteratorToSeqs(it)
	want := []int64{2, 4, 7}
	assert.Equal(t, want, got)
}

func Test_ThrottledIterator(t *testing.T) {
	it := NewChanIterator(makeLogsChan(1, 3, 5, 7, 7, 7, 7, 8))
	tit := NewThrottledIterator(it, 1, 1*time.Second)
	got := convertIteratorToSeqs(tit)
	want := []int64{1, 3, 5, 7, 7, 7, 7, 8}
	assert.Equal(t, want, got)
}
