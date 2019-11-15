package logmerger

import (
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
	inputc  chan Log
	outputc chan Log

	sleepInterval time.Duration
	step          int64
	stats         LogConsumerStats
}

var currentMockToken int64 = 1000

var _ LogConsumer = &FakeLogConsumer{}

func NewFakeLogConsumer(inputc chan Log) *FakeLogConsumer {
	return &FakeLogConsumer{
		inputc:  inputc,
		outputc: make(chan Log, 1024),

		sleepInterval: 1 * time.Second,
		stats:         LogConsumerStats{},
	}
}

func (c *FakeLogConsumer) Run(position int64, quitc chan struct{}) {
	for {
		select {
		case <-quitc:
			return
		case lg := <-c.inputc:
			c.stats.LastMessageTimestamp = time.Now().Unix()
			c.stats.MessagesTotal++
			if lg.kind == "commit" {
				c.stats.CommitsTotal++
			} else if lg.kind == "prepare" {
				c.stats.PreparesTotal++
			}
			c.output(lg, quitc)
		}
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
