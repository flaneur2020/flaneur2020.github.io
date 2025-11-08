https://medium.com/@jimmysong/understanding-segwit-block-size-fd901b87c9d4
https://blockchain.iethpay.com/transaction-malleability.html
https://bitcoincore.org/zh_CN/2016/01/26/segwit-benefits/
https://medium.com/@twedusuck/%E6%AF%94%E7%89%B9%E5%B9%A3-%E4%BB%A5%E5%A4%AA%E5%9D%8A%E7%9A%84%E4%B8%80%E4%BA%9B%E5%95%8F%E9%A1%8C%E4%BB%8B%E7%B4%B9-%E4%BA%8C-bc06a5e7f8fc
https://www.zhihu.com/question/264244976/answer/278623217

动机
- 解决交易延展性攻击(transaction malleability)问题；
transaction malleability 问题
- tx 做签名时，并没有覆盖 tx 相关的所有数据（比如位于 txin 中的 scriptSig）；
- 这样一来，bitcoin 网络中的一个节点能任意修改你发送的交易 tx 的哈希值；
	- 攻击者仅能修改哈希值，交易中的 txout 不可改变，你仍能正确地转账到正确的地址上；
- 然而，这导致未确认的 tx 的哈希值是不可信任的；
transaction melleability 的利用
- 2014 年有人利用这一漏洞攻击 mtgox：
	- 先从 mtgox 申请提现，同时发动自己的节点修改交易的 tx 哈希；
	- 如果 mtgox 通过 tx 哈希去查询确认数，发现没有得到确认而重发时，攻击者就多拿了一分钱；
- 预防的机制就是，不要将 tx 哈希当做交易确认的查询条件；
segwit 的方案：SegWit replaces the 1MB block-size limit with a 4MB block-weight limit
- 注意扩大的是 4mb block weight （载荷）而不是 block size（大小）；
- 将交易区分为账目进出（地址、金额）和见证（witness，即签名）两部分：
	- 如果只关心每个账目的余额，只保存账目部分即可，至于包括见证信息的全量 block chain 只需要矿工等少数保留；
	- 中本聪的原始设计中并没有区分两者，会对 sigScript 也做哈希，导致 txid 可变；
	- segwit 的设计中，将只根据账目计算确定的 txid；
- segwit 用户在交易时，会将币传送给有别于传统 bitcoin address 的地址，当要使用这些比特币的时候，其签署 (即见证) 并不会加入交易 ID 的哈希计算；
	- https://github.com/jl2012/bips/blob/segwit-address/bip-segwitaddress.mediawiki
segwit 的好处
- 解决交易延展性问题：
	- 钱包管理者可以根据 tx 哈希来跟踪交易的确认状况了，提高了安全性；
	- 可以花费未确认的交易：对于从交易所里提取的币，可以直接引用该 tx 哈希，在确认前花掉了；
- 可以通过 soft fork 方式增大区块容量：旧客户端看不到区块的见证部分，仍以为 block size 为 1mb；
- 钱包实现上可以更轻量：不需要下载见证数据；
segwit 的 soft fork：
- 旧的 bitcoin 客户端会拒绝大于 1mb 的区块；
- 不通过硬分叉的方式扩大区块大小，而是从区块（仍 1mb）中分离出附加信息（3mb）；
- soft fork 的操作是，直到 95% 的节点都部署到 segwit 之后才激活 segwit 特性；
争议
- 一部分人 argue 说 segwit 的扩容仍不能满足 bitcoin 网络当前的需求；
- segwit 的作者也是 blockstream 公司的成员，而它们侧重于开发自己的侧链产品；
- 主要的 ideological 方面争议在于，方案并没有做到在保留 bitcoin 去中心化特性的前提下，提供更好的扩展性；
- segwit2x 难产：跟 segwit 其实没关系，搞硬分叉扩容大区块的路线，Barry Silbert 跟比特大陆搞了一个纽约共识；