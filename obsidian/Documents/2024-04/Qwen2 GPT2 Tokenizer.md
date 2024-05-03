有 tokens 文件和 merges 文件。

merges 文件每行用空格区分。

bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

bpe_ranks 的类型大约是 Map<(String, String), usize>

找出来 bpe_rank 最小的 bigram: (String, String)

## qwen2 tiktok.h

### 入口

```
		tiktoken(
			ankerl::unordered_dense::map<std::string, int> encoder,
			ankerl::unordered_dense::map<std::string, int> special_encoder,
			const std::string &pattern
```

#### Q：encoder 是什么？

大约等于 vocabs，每个 token 对应一个 token id。
#### Q：special_encoder 是什么？

```C++
  std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>", "<|im_end|>"};
  char buffer[14];
  for (size_t i = 0; i < 205; i++) {
    snprintf(buffer, 14, "<|extra_%zu|>", i);
    special_tokens_s.push_back(buffer);
  }
  size_t encoder_size = encoder.size();
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  special_tokens.reserve(special_tokens_s.size());
  for (size_t i = 0; i < special_tokens_s.size(); i++) {
    special_tokens[special_tokens_s[i]] = encoder_size + i;
  }
```

special_encoder 似乎是 `<|endoftext|>` 这些，然后 `<|extra_%zu|>` 还不知道干嘛用的。

special_encoder 的 token id 会追加到 vocabs 的最后面。
### 编码过程

encode 是编码的入口，它会接着调用 `_encode_native`。

先调用 `split_with_allowed_special_token` 按 special tokens 拆分。

针对非 special token 的部分，执行 `byte_pair_encode`。

对于 special token，查 special token 表，得到 special token 对应的 token id。

`byte_pair_encode` 

```C++
	static auto byte_pair_encode(
		const std::string &piece,
		const ankerl::unordered_dense::map<std::string, int> &ranks) -> std::vector<int>
	{
		if (piece.size() == 1)
		{
			return {ranks.at(piece)};
		}

		auto func = [&piece, &ranks](int start, int stop) -> int
		{
			std::string key = piece.substr(start, stop - start);
			return ranks.at(key);
		};

		return _byte_pair_merge(piece, ranks, func);
	}
```

里面的这个 ranks 其实是 token 到 token id 的映射表。

`_byte_pair_merge` 这个函数拿一个回调函数，似乎会遍历 piece 中的某段字符串，找在里面是否存在。

#### Q: ranks 等于 tokens table，那么 merges 在哪里？


## `_byte_pair_merge` 本体

```C++
	static auto _byte_pair_merge(
		const std::string &piece,
		const ankerl::unordered_dense::map<std::string, int> &ranks,
		std::function<int(int, int)> func) -> std::vector<int>
	{
		std::vector<std::pair<int, int>> parts;
		parts.reserve(piece.size() + 1);
		for (auto idx = 0U; idx < piece.size() + 1; ++idx)
		{
			parts.emplace_back(idx, std::numeric_limits<int>::max());
		}

		auto get_rank = [&piece, &ranks](
							const std::vector<std::pair<int, int>> &parts,
							int start_idx,
							int skip) -> std::optional<int>
		{
			if (start_idx + skip + 2 < parts.size())
			{
				auto s = parts[start_idx].first;
				auto e = parts[start_idx + skip + 2].first;
				auto key = piece.substr(s, e - s);
				auto iter = ranks.find(key);
				if (iter != ranks.end())
				{
					return iter->second;
				}
			}
			return std::nullopt;
		};

		for (auto i = 0U; i < parts.size() - 2; ++i)
		{
			auto rank = get_rank(parts, i, 0);
			if (rank)
			{
				assert(*rank != std::numeric_limits<int>::max());
				parts[i].second = *rank;
			}
		}

		while (true)
		{
			if (parts.size() == 1)
				break;

			auto min_rank = std::make_pair<int, int>(std::numeric_limits<int>::max(), 0);
			for (auto i = 0U; i < parts.size() - 1; ++i)
			{
				auto rank = parts[i].second;
				if (rank < min_rank.first)
				{
					min_rank = {rank, i};
				}
			}

			if (min_rank.first != std::numeric_limits<int>::max())
			{
				auto i = min_rank.second;
				auto rank = get_rank(parts, i, 1);
				if (rank)
				{
					parts[i].second = *rank;
				}
				else
				{
					parts[i].second = std::numeric_limits<int>::max();
				}
				if (i > 0)
				{
					auto rank = get_rank(parts, i - 1, 1);
					if (rank)
					{
						parts[i - 1].second = *rank;
					}
					else
					{
						parts[i - 1].second = std::numeric_limits<int>::max();
					}
				}

				parts.erase(parts.begin() + (i + 1));
			}
			else
			{
				break;
			}
		}
		std::vector<int> out;
		out.reserve(parts.size() - 1);
		for (auto i = 0U; i < parts.size() - 1; ++i)
		{
			out.push_back(func(parts[i].first, parts[i + 1].first));
		}
		return out;
	}

```

流程：

先初始化一个长度等于 piece 的 `parts: Vec<(usize, usize)>`。

定义一个 get_rank 的 lambda。

针对每个 charactor，找出来它对应的 rank，放在 parts 表中。

每次合并优先合并 rank 越低的。

这么合并到没有可以合并的为止。


---

## tokenizer-rs

```rust
pub fn split_on_bpe_pairs<F>(
    token: TokenRef<'_>,
    bpe_function: F,
    bpe_ranks: &BpePairVocab,
    cache: &BpeCache,
    as_bytes: bool,
) -> Vec<Token>
where
    F: Fn(&str, &BpePairVocab) -> (Vec<String>, Vec<usize>),
{
    let mut tokens: Vec<Token> = Vec::new();
    let text: String;
    let reference_offsets_placeholder: Vec<OffsetSize>;
    let (text, reference_offsets) = if as_bytes {
        reference_offsets_placeholder = bytes_offsets(token.text)
            .iter()
            .map(|&pos| token.reference_offsets[pos])
            .collect();
        text = token
            .text
            .as_bytes()
            .iter()
            .map(|v| BYTES_TO_UNICODE.get(v).unwrap())
            .collect();
        (text.as_str(), reference_offsets_placeholder.as_slice())
    } else {
        (token.text, token.reference_offsets)
    };

        let (bpe_output, char_counts) = bpe_function(text, bpe_ranks);
        if let Ok(mut cache) = cache.try_write() {
            cache.insert(text.to_owned(), (bpe_output.clone(), char_counts.clone()));
        }
        let mut start = 0;
        for (idx, (sub_token, &char_count)) in bpe_output.iter().zip(char_counts.iter()).enumerate()
        {
            tokens.push(Token {
                text: sub_token.clone(),
                offset: Offset {
                    begin: reference_offsets[start],
                    end: reference_offsets[start + char_count - 1] + 1,
                },
                reference_offsets: reference_offsets[start..start + char_count].to_vec(),
                mask: {
                    if bpe_output.len() > 1 {
                        if idx == 0 {
                            Mask::Begin
                        } else {
                            Mask::Continuation
                        }
                    } else {
                        Mask::None
                    }
                },
            });
            start += char_count;
        }

    tokens
}
```

bytes_to_unicode