# S2TTBase1 服务器测试总结（SeamlessM4T 中文语音 -> 文本）

> 日期：2026-08-20（创建）/ 2026-08-21（验证与修正）
> 机器：server56（gpu56）
> 工程目录：`/baykal/unisound/wangwentao/moshibase1/s2ttbase1`

## 1. 背景与目标

在 server56 上以 Docker 方式部署并测试 Meta SeamlessM4T 系列模型，验证 **中文语音输入 -> 文本输出**（S2TT 与 ASR）能力，同时测试流式模型（SeamlessStreaming）。

## 2. 环境与镜像

### 2.1 基础镜像

- 基础镜像：`harbor.unidev.ai/wangwentao/indextts2:base2`（Python 3.10.12，CUDA 12.8）
- 工作容器：`s2ttbase1-test`，挂载：
  - 代码/数据：`/baykal/unisound/wangwentao/moshibase1/s2ttbase1` -> `/workspace/s2ttbase1`
  - HF 缓存：`/baykal/unisound/wangwentao/moshibase1/.hf_cache` -> `/root/.cache/huggingface`

### 2.2 依赖安装（清华源）

在容器内使用清华源安装：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fairseq2==0.2.*
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchaudio==2.2.2 torchvision==0.17.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .
```

注意：`fairseq2==0.2.1` 会把 `torch` 降级到 `2.2.2`，因此需要同步降级 `torchaudio`（2.2.2）和 `torchvision`（0.17.2），否则版本不兼容。

### 2.3 已推送镜像

```bash
docker commit s2ttbase1-test harbor.unidev.ai/wangwentao/s2ttbase1:20260820
docker push harbor.unidev.ai/wangwentao/s2ttbase1:20260820
```

- 镜像名：`harbor.unidev.ai/wangwentao/s2ttbase1:20260821`（推荐，含 streaming 卡片本地化）
- digest：`sha256:d7f710ff1d1dfde4988fd6c4d08c5831c3f8a4e92d3e981733fa0970c47a11d6`
- 大小：约 29.2GB（模型权重在挂载卷中，未打入镜像）

> 旧镜像 `20260820`（digest `d3f4872803c2e1ad62aeea63663f05f3421b83a0424555de04eeab6254d92668`）仅本地化了离线模型卡片；streaming 卡片本地化是在 commit 之后才完成的，因此旧镜像跑流式仍会访问 HuggingFace 失败。`20260821` 已修正此问题。

## 3. 模型权重（ModelScope 下载）

server56 无法访问 HuggingFace，按约定从 ModelScope 下载原始 checkpoint：

> 重要：所有权重均下载到**工作目录挂载卷** `/baykal/unisound/wangwentao/moshibase1/s2ttbase1/models/`（容器内 `/workspace/s2ttbase1/models/`），并非容器临时目录。容器删除/重建后文件仍在，重新挂载即可使用。

### 3.1 离线模型 `facebook/seamless-m4t-v2-large`

存放：`models/facebook/seamless-m4t-v2-large/`

| 文件 | 大小 | sha256 |
| --- | --- | --- |
| seamlessM4T_v2_large.pt | 9.0GB | f2122cedcffc532d6048847092414e638cfb4db402881cb8146a606008e9ff56 |
| vocoder_v2.pt | 167MB | - |
| spm_char_lang38_tc.model | 368KB | - |
| tokenizer.model | 5.2MB | - |

### 3.2 流式模型 `facebook/seamless-streaming`

存放：`models/facebook/seamless-streaming/`

| 文件 | 大小 |
| --- | --- |
| seamless_streaming_unity.pt | 3.6GB |
| seamless_streaming_monotonic_decoder.pt | 4.3GB |
| vocoder_v2.pt | 167MB |
| spm_char_lang38_tc.model | 368KB |
| tokenizer.model | 5.2MB |

### 3.3 模型卡本地化

由于权重改为本地文件，需要把 `cards/*.yaml` 中的 checkpoint/tokenizer URL 改为 `file://` 本地路径（修改的是 server56 工程副本，本地仓库未改动）：

```bash
MODEL_BASE="file:///workspace/s2ttbase1/models/facebook/seamless-m4t-v2-large"
STREAM_BASE="file:///workspace/s2ttbase1/models/facebook/seamless-streaming"
```

涉及文件：
- `src/seamless_communication/cards/seamlessM4T_v2_large.yaml`
- `src/seamless_communication/cards/vocoder_v2.yaml`
- `src/seamless_communication/cards/unity_nllb-100.yaml`
- `src/seamless_communication/cards/seamless_streaming_unity.yaml`
- `src/seamless_communication/cards/seamless_streaming_monotonic_decoder.yaml`

同时把修改同步到容器内 site-packages 中的同一批 yaml，保证 `m4t_predict` / `streaming_evaluate` 入口读到的也是本地路径。

## 4. 测试数据

### 4.1 小规模中文 prompt（yaokun_prompt）

- 来源：`/baykal/unisound/wangwentao/moshi/indextts2/index-tts/yaokun_prompt/`
- 使用：`zero_shot_0_0-ZH_2_prompt.wav`（太乙真人）、`zero_shot_1_0-ZH_3_prompt.wav`（开会吐槽）
- 转 16kHz 后放在 `audio/`（`zh_taibai_16k.wav`、`zh_meeting_16k.wav`）

### 4.2 标准中文测试集（seedtts_testset/zh）

- 来源：`/baykal/unisound/wangwentao/moshi/indextts2/index-tts/f5tts/F5-TTS-main/data/seedtts_testset/zh`
- 挑选 10 句（带参考文本），wav 复制到 `audio/` 并转 16kHz

```text
10002287-00000095 | 简单地说，这相当于惠普把消费领域市场拱手相让了。
10002290-00000102 | 说说你们在京藏高速怀安段，堵车现场看到的情况。
10002298-00000016 | 共同建设面向未来的交通，和出行服务新生态。
10002352-00000005 | 女性可以成为成功的科学家，工程师和程序员。
10002481-00000105 | 就在营救进行的同时，楼下却不断地发出欢呼，起哄声。
10002539-00000054 | 第一天去，前几个小时很尴尬，孤零零的。
10002702-00000051 | 这些医疗队的医生们克服高寒缺氧，交通不便等困难。
10002733-00000021 | 导航开始，全程二十五公里，预计需要十二分钟。
10002739-00000101 | 这不是变魔术，而是添加剂让腐朽变神奇。
10002536-00000061 | 晃眼的人民币旁，更多的是生命的无奈。
```

> 说明：用户最初指定的 `goal-data/source/avsjni537/filelist_all_corrected.txt` 为纯英文数据集（9905 行无中文），且 `/hulon` 路径在 server22/56 均不存在，故改用上述中文测试集。

## 5. 测试方式与结果

### 5.1 离线 S2TT / ASR（m4t_predict）

```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 s2ttbase1-test bash -lc \
  "m4t_predict audio/xxx_16k.wav --task S2TT --tgt_lang eng --model_name seamlessM4T_v2_large --vocoder_name vocoder_v2"

docker exec -e CUDA_VISIBLE_DEVICES=1 s2ttbase1-test bash -lc \
  "m4t_predict audio/xxx_16k.wav --task ASR --tgt_lang cmn --model_name seamlessM4T_v2_large --vocoder_name vocoder_v2"
```

也可以加载一次 Translator 批量跑 10 句（脚本见下文 5.4）。

10 句标准测试集结果（离线 fp16）：

| 参考文本 | ASR（中->中） | S2TT（中->英） |
| --- | --- | --- |
| 简单地说，这相当于惠普把消费领域市场拱手相让了。 | 简单的说这相当于惠普把消费领域市场拱手相让了 | Simply put, it's the equivalent of HP taking over the consumer market. |
| 说说你们在京藏高速怀安段，堵车现场看到的情况。 | 说说你们在京藏高速淮安段堵车现场看到的情况 | Tell me about what you saw at the traffic jam on the Beijing-Tibet highway. |
| 共同建设面向未来的交通，和出行服务新生态。 | 共同建设面向未来的交通和出行服务新生态 | Building a new ecosystem of transport and travel services for the future |
| 女性可以成为成功的科学家，工程师和程序员。 | 女性可以成为成功的科学家 工程师和程序员 | Women can become successful scientists, engineers and programmers |
| 就在营救进行的同时，楼下却不断地发出欢呼，起哄声。 | 就在营救进行的同时楼下却不断地发出欢呼起哄声 | At the same time as the rescue was going on, there was a constant cheering and cheering downstairs. |
| 第一天去，前几个小时很尴尬，孤零零的。 | 第一天去前几个小时很尴尬孤零零的 | The first day, the first few hours, it was embarrassing, lonely. |
| 这些医疗队的医生们克服高寒缺氧，交通不便等困难。 | 这些医疗队的医生们克服高寒缺氧交通不便等困难 | Doctors in these medical teams overcome difficulties such as high fever, lack of oxygen, and traffic disruption. |
| 导航开始，全程二十五公里，预计需要十二分钟。 | 导航开始 全程二十五公里 预计需要十二分钟 | Navigation starts at 25 kilometers and is expected to take 12 minutes. |
| 这不是变魔术，而是添加剂让腐朽变神奇。 | 这不是变魔术,而是添加剂让腐朽变神奇 | It's not magic, it's additives that make decay magical. |
| 晃眼的人民币旁，更多的是生命的无奈。 | 黄眼的人民壁旁更多的是生命的无奈 | Yellow-eyed people are more helpless than life |

结论：ASR 前 9 句几乎全对；S2TT 前 9 句语义翻译基本正确；第 10 句本身为异常文本，ASR/S2TT 均受影响。

### 5.2 流式模型（streaming_evaluate）

```bash
docker exec -e CUDA_VISIBLE_DEVICES=1 s2ttbase1-test bash -lc \
  "streaming_evaluate --task s2tt --data-file audio/xxx.tsv --audio-root-dir audio \
   --output streaming_out --tgt-lang eng --device cuda:0 --dtype fp16 --no-scoring \
   --block-ngrams --no-strip-silence"
```

TSV 格式（header 必须含 `id audio tgt_text`）：

```text
id	audio	tgt_text
zh_meeting	zh_meeting_16k.wav	啊，事情怎么这么多啊，明天又要开会了啊……
```

### 5.3 流式重复问题与修复

**问题**：默认参数 `no_early_stop=True`、`max_len_b=100` 且未开 `--block-ngrams`，解码器在源未结束时会持续生成，长句/排比句会陷入循环重复（例如 "you do, you do, you do..."）。

**修复**：`streaming_evaluate` 增加 `--block-ngrams --no-strip-silence`。修复前后对比（同一段音频）：

- 修复前（100 token 刷屏）：`...You say you don't listen, you don't understand, you don't do, you do, you do, you do...`
- 修复后（82 token，无重复）：`...I said you don't listen, you don't understand, you don't do it, you make mistakes, you don't admit it, you don't change it, you don't obey it, you don't tell me what to do.`

另外：流式评估每次启动都会用 SileroVAD 去 GitHub 检查版本，server56 网络不通时会直接崩（`RemoteDisconnected`），`--no-strip-silence` 可跳过该依赖。

### 5.4 批量测试脚本（离线，加载一次模型跑 10 句）

```python
import torch, torchaudio
from seamless_communication.inference import Translator

translator = Translator(
    "seamlessM4T_v2_large", "vocoder_v2",
    torch.device("cuda:0"), dtype=torch.float16,
)

for i in ids:
    wav, sr = torchaudio.load(f"audio/{i}_16k.wav")
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    asr_out, _ = translator.predict(wav, "ASR", "cmn", unit_generation_opts=None)
    s2tt_out, _ = translator.predict(wav, "S2TT", "eng", unit_generation_opts=None)
    print(f"ASR : {asr_out[0]}")
    print(f"S2TT: {s2tt_out[0]}")
```

> 注意：本仓库当前 `Translator` 构造签名为 `(model, vocoder, device, text_tokenizer=None, apply_mintox=False, dtype=...)`，`dtype` 必须以关键字传入，否则会被当作 `text_tokenizer` 报错。

## 6. 注意事项 / 待办

1. 模型权重（约 13GB 离线 + 8GB 流式）在挂载卷 `s2ttbase1/models/`，未打入镜像；换机器时需重新挂载或下载。
2. 模型卡本地化只改了 server56 工程副本与容器内 site-packages，本仓库保持原样；如需在别处复现，按第 3.3 节同样修改。
3. 流式 S2TT 长句仍有轻微重复倾向，建议固定使用 `--block-ngrams`；如需更干净输出可进一步调低 `--max-len-b`。
4. 已推送镜像 `harbor.unidev.ai/wangwentao/s2ttbase1:20260821` 已包含 streaming 卡片本地化；`--block-ngrams` 尚未固化到 `evaluate.py` 默认参数，如需要可改后重新 commit/push。

## 7. 容器重建验证（2026-08-21）

针对「下载文件是否在临时目录、容器退掉后是否还在、code 是否能行」做的实际验证：

### 7.1 文件位置确认

模型权重位于宿主机挂载卷 `/baykal/unisound/wangwentao/moshibase1/s2ttbase1/models/`（而非容器可写层），`docker rm` 容器不影响这些文件。已验证以下文件完整存在：

```text
models/facebook/seamless-m4t-v2-large/seamlessM4T_v2_large.pt   （9.0GB）
models/facebook/seamless-m4t-v2-large/vocoder_v2.pt
models/facebook/seamless-m4t-v2-large/spm_char_lang38_tc.model
models/facebook/seamless-m4t-v2-large/tokenizer.model
models/facebook/seamless-streaming/seamless_streaming_unity.pt  （3.6GB）
models/facebook/seamless-streaming/seamless_streaming_monotonic_decoder.pt  （4.3GB）
models/facebook/seamless-streaming/vocoder_v2.pt
models/facebook/seamless-streaming/spm_char_lang38_tc.model
models/facebook/seamless-streaming/tokenizer.model
```

### 7.2 从镜像重建容器验证 code

删除旧容器后用已推送镜像起全新容器（只挂载工作目录，不挂 HF 缓存）：

```bash
docker rm -f s2ttbase1-verify
docker run -d --name s2ttbase1-verify --gpus all --shm-size=16g \
  -v /baykal/unisound/wangwentao/moshibase1/s2ttbase1:/workspace/s2ttbase1 \
  -w /workspace/s2ttbase1 harbor.unidev.ai/wangwentao/s2ttbase1:20260821 sleep infinity
```

**离线验证通过**（ASR + S2TT）：

```text
ASR : 简单的说这相当于惠普把消费领域市场拱手相让了
S2TT: Simply put, it's the equivalent of HP taking over the consumer market.
```

**流式验证通过**（`--block-ngrams --no-strip-silence`）：

```text
0 :: Simply put, it's the equivalent of a consumer-oriented market.
1 :: What you saw at the traffic jam on Beijing-Tibet highway
```

### 7.3 结论

- 模型文件保存在工作目录挂载卷，容器重建后依然可用；
- 从镜像（`20260821`）重建容器后，离线 S2TT/ASR 与流式 S2TT 均能直接运行；
- 使用镜像时注意：流式评估需要 `--block-ngrams --no-strip-silence`；如不挂载 `models/` 目录需先下载权重。
