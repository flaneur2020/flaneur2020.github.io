package logmerger

import (
	"math/rand"
	"sync/atomic"
	"time"
)

type LogConsumerStats struct {
	MessagesTotal        uint64 `json:"messagesTotal"`
	PreparesTotal        uint64 `json:"preparesTotal"`
	CommitsTotal         uint64 `json:"commitsTotal"`
	LastMessageTimestamp int64  `json:"lastMessageTimestamp"`
}

// LogConsumer 接口对接消息源，如 Kafka、Binlog 等。
// LogConsumer 遇到任何错误皆应从上次点位开始继续消费
type LogConsumer interface {
	Run(poistion int64, quitc chan struct{})
	Close()
	Iterator() LogIterator
	Stats() LogConsumerStats
}

type FakeLogConsumer struct {
	outputc chan Log

	sleepInterval time.Duration
	step          int64
	stats         LogConsumerStats
}

var currentMockToken int64 = 1000

var _ LogConsumer = &FakeLogConsumer{}

func NewFakeLogConsumer() *FakeLogConsumer {
	return &FakeLogConsumer{
		outputc: make(chan Log, 512),

		sleepInterval: 1 * time.Second,
		stats:         LogConsumerStats{},
	}
}

func (c *FakeLogConsumer) Run(position int64, quitc chan struct{}) {
	for {
		select {
		case <-quitc:
			return
		default:
		}

		c.stats.LastMessageTimestamp = time.Now().Unix()
		c.stats.MessagesTotal++

		prepareToken := c.nextToken()
		lg := Log{
			kind:         "prepare",
			data:         []byte("mock message"),
			prepareToken: prepareToken,
		}
		c.output(lg, quitc)

		c.stats.PreparesTotal++
		c.randomSleep()

		lg = Log{
			kind:         "commit",
			prepareToken: prepareToken,
			commitToken:  c.nextToken(),
		}
		c.output(lg, quitc)

		c.stats.CommitsTotal++
		c.randomSleep()
	}
}

func (c *FakeLogConsumer) output(lg Log, quitc chan struct{}) {
	select {
	case <-quitc:
	case c.outputc <- lg:
	}
}

func (c *FakeLogConsumer) Close() {
	close(c.outputc)
}

func (c *FakeLogConsumer) Stats() LogConsumerStats {
	return c.stats
}

func (c *FakeLogConsumer) Iterator() LogIterator {
	return NewCommitedLogIterator(NewChanIterator(c.outputc))
}

func (c *FakeLogConsumer) nextToken() int64 {
	maxGap := int64(10)
	return atomic.AddInt64(&currentMockToken, rand.Int63()%maxGap+1)
}

func (c *FakeLogConsumer) randomSleep() {
	rnd := rand.Int63() % 5
	time.Sleep(time.Duration(rnd) * time.Millisecond)
}
