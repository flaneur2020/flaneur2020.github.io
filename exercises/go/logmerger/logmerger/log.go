package logmerger

type Log struct {
	kind         string
	data         []byte
	prepareToken int64
	commitToken  int64
}

