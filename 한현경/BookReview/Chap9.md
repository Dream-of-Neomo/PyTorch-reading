๐ <b> [9์ฅ] ์๊ณผ ์ธ์ ์ด๊ธฐ๊ธฐ ์ํ ํ์ดํ ์น ํ์ฉ</b> (โ'โก'โ)

### 9.2 ๋๊ท๋ชจ ํ๋ก์ ํธ ์ค๋น
- <b>ํ์์ ํ๋ถ CT ์ค์บ์ ์๋ ฅ์ผ๋ก ๋ฐ์์ ์์ฑ ์ข์์ ํ์งํ๋ ๋ฅ๋ฌ๋ ํ๋ก์ ํธ</b>
- ์ค๋น์ฌํญ
    - ํ๋ก์ ํธ๋ฅผ ์งํํ๋ ค๋ฉด ๋ชจ๋ธ ์ํคํ์ฒ ์ค๊ณ์ ์์ ์ฌ์ฉํ  ๋ฐ์ดํฐ์ ๋ํด ์์์ผ ํ๋ค.
    - ํ๋ก์ ํธ๋ฅผ ์ง์  ์คํํ๊ธฐ ์ํด์๋ `์ต์ 8GB ์ด์์ RAM์ ๊ฐ์ถ GPU`๊ฐ ํ์ํ๋ค. (GPU๊ฐ ์๋ ๊ฒฝ์ฐ Chap.14์ ์ฌ์  ํ๋ จ๋ ๋ชจ๋ธ ์ฌ์ฉ)
    ๊ทธ๋ฆฌ๊ณ  ์๋ณธ ํ๋ จ ๋ฐ์ดํฐ, ์บ์ ๋ฐ์ดํฐ, ํ๋ จ๋ ๋ชจ๋ธ์ ์ ์ฅํ๊ธฐ ์ํด `220GB์ ๋์คํฌ ๊ณต๊ฐ`์ด ํ์ํ๋ค.
- ๋ฌธ์ ๋ฅผ ํด๊ฒฐํ๋ ํ๋
    - `Divide and Conquer` : ๋ฌธ์ ๋ฅผ ์์ ์กฐ๊ฐ์ผ๋ก ๋๋ ์ ์๊ฐํ๋ค.
    - `๋ฐฐ๊ฒฝ์ง์` ์๊ธฐ : ์ํ ๋ถ์ผ์ ๋ํ ํต์ฐฐ๋ ฅ์ ๊ฐ์ ธ์ผ ํ๋ค.

### 9.3 CT ์ค์บ์ด๋
- ์ฑ์ ํ๋ก์ ํธ์์๋ ์ฃผ๋ก CT์ค์บ ๋ฐ์ดํฐ๋ฅผ ๋ค๋ฃฌ๋ค.
- `CT์ค์บ`์ ๋จ์ผ ์ฑ๋์ 3์ฐจ์ ๋ฐฐ์ด๋ก ํํ๋๋ 3์ฐจ์ ์์ค๋ ์ด์ด๋ค. (๋ง์น ํ๋ฐฑ PNG ์ด๋ฏธ์ง๋ฅผ ์ฐจ๊ณก์ฐจ๊ณก ์์๋ ํํ)
- `๋ณต์(voxel)`์ด๋?
    - `VOlumetric piXEL(์ฉ์  ํฝ์)`
    - ๋ฉด์ ์ด ์๋ ๊ณต๊ฐ ์ฉ์ ์ ๋ด๊ณ  ์์ผ๋ฉฐ, ๋ฐ์ดํฐ ํ๋๋ฅผ ํํํ๊ธฐ ์ํด 3์ฐจ์ ๊ฒฉ์๋ก ๋ฐฐ์ด๋๋ค. ๊ฐ ์ฐจ์ ๋ด์์๋ ๊ฑฐ๋ฆฌ๋ฅผ ์ธก์ ํ  ์ ์๋ค. ํต์์ ์ผ๋ก ๋ณต์์ ์ ์ก๋ฉด์ฒด์ง๋ง ์ด ์ฅ์์ ๋ค๋ฃจ๋ ๋ณต์์ ์ง์ก๋ฉด์ฒด์ด๋ค.
    - ์ํ ๋ถ์ผ ์ธ์๋ 2์ฐจ์ ์ด๋ฏธ์ง๋ก ์ฌ๊ตฌ์ฑ๋ 3์ฐจ์ ์ฅ๋ฉด, ์์จ์ฃผํ ์๋์ฐจ์ ๋ผ์ด๋ค(LIDAR, Light Detection and Ranging) ๋ฐ์ดํฐ์์๋ ๋ณต์์ด ์ฐ์ธ๋ค.
- CT์ค์บ ๋ด ๋ณต์์ ํด๋น ์์น์ ์๋ ๋ฌผ์ฒด์ ํ๊ท  ์ง๋ ๋ฐ๋๋ฅผ ๋ํ๋ด๋ ์ซ์ ๊ฐ์ ๊ฐ์ง๋ค.
    - ๋ฐ์ดํฐ๋ค์ ์๊ฐํํ๋ฉด ๋ฐ๋๊ฐ ๋์ ์กฐ์ง(๋ผ๋ ๊ธ์ ์ํ๋ํธ)์ ํ์์, ๋ฐ๋๊ฐ ๋ฎ์ ์กฐ์ง(๊ณต๊ธฐ, ํ์กฐ์ง)์ ๊ฒ์์์ผ๋ก ๋ณด์ด๊ณ , ์ง๋ฐฉ์ด๋ ์กฐ์ง์ ๋ค์ํ ๋ฐ๊ธฐ์ ํ์์ผ๋ก ๋ํ๋๋ค. 
- CT vs. X-ray
    - X-ray์ ๊ฒฝ์ฐ 3์ฐจ์ ๊ฐ๋๋ฅผ 2์ฐจ์ ํ๋ฉด์ ํฌ์ํ ๋ฐ๋ฉด CT์ค์บ์ ๋ฐ์ดํฐ์ 3์ฐจ์ ํํ๋ฅผ ๋ณด์กดํ๋ค.
    - CT๋ X-ray์ ๋ฌ๋ฆฌ ๋ฐ์ดํฐ๊ฐ digital format์ด๋ค. ์ฆ, ์ค์บ์ ์๋ณธ ์ถ๋ ฅ์ ์ก์์ผ๋ก ์์๋ณผ ์ ์๋ ํํ๊ฐ ์๋๋ผ์ ์ปดํจํฐ๋ฅผ ์ฌ์ฉํด ์ฌํด์ํด์ผ ํ๋ค. ์ค์บ ์ CT ์ค์บ๋์ ์ค์ ์ ๋ฐ๋ผ ์ถ๋ ฅ ๋ฐ์ดํฐ๊ฐ ๋ฌ๋ผ์ง๋ค.

### 9.4 ํ๋ก์ ํธ: ์๋ํฌ์๋ ํ์ ์ง๋จ๊ธฐ
- `๊ฒฐ์ `์ด๋?
    - ์ฆ์ํ๋ ์ธํฌ๋ก ์ด๋ค์ง ์กฐ์ง ๋ฉ์ด๋ฆฌ๋ฅผ ์ข์(tumor)์ด๋ผ๊ณ  ํ๋ค. ์ข์์ ์์ฑ(benign) ํน์ ์์ฑ(malignant)๋ก ๋๋๋ฉฐ, ์์ฑ์ผ ๊ฒฝ์ฐ ์์ด๋ผ๊ณ ๋ ํ๋ค. ํ์ ์๋ ์์ ์ข์์ ํ ๊ฒฐ์ (lung nodule)์ด๋ผ๊ณ  ํ๋ค. ํ ๊ฒฐ์ ์ 40%๋ ์์ฑ์ธ ๊ฒ์ผ๋ก ์๋ ค์ง๋ค. ๋ฐ๋ผ์ ์ต๋ํ ์ด๊ธฐ์ ์ก์๋ด๋ ๊ฒ์ด ์ค์ํ๋ฐ, ์ด ํ๋ก์ ํธ์์์ฒ๋ผ ์ด๋ฅผ ์ํ ์์ ์ด๋ฏธ์ง๋ก ๋ฐ๊ฒฌํ  ์ ์๋ค. 
    - ์์ ํญ์ ๊ฒฐ์ ์ ํํ๋ฅผ ๋ค๋ค.(๋ฐ๋ผ์ classifier๋ ๋ชจ๋  ์กฐ์ง์ ์์๋ณผ ํ์ ์์ด ๊ฒฐ์ ์๋ง ์ง์คํ๋ฉด ๋๋ค.)

- ํ๋ถ CT์ค์บ ์กฐ์ฌ์์ ํ์์๊ฒ ํ์ ์ง๋จ์ ๋ด๋ฆฌ๋ ๊ณผ์ ์ `์ด 5๊ฐ์ ๋จ๊ณ`๋ก ๋๋ ์ ์๋ค. 
    1. `Data Loading` : ์๋ณธ CT์ค์บ ๋ฐ์ดํฐ๋ฅผ ํ์ดํ ์น์์ ์ฌ์ฉํ  ์ ์๋ ํํ๋ก ์ฝ์ด๋ค์ธ๋ค. 
    2. `Segmentation` : ํ์ดํ ์น๋ฅผ ์ฌ์ฉํ์ฌ ํ์ ์ ์ฌ์  ๊ฒฐ์ ์ ํด๋นํ๋ ๋ณต์์ ์ฐพ๋๋ค. ์ผ์ข์ ํํธ๋งต์ ์์ฑํ๋ ๊ณผ์ ์ด๋ค. ๊ด์ฌ์๋ ํด๋ถํ์  ๋ถ์(์, ๊ฐ ๋ฑ ํ์๊ณผ๋ ๋ฌด๊ดํ ๋ถ์)๋ฅผ ๋ฌด์ํ๊ณ  ํ ๋ด๋ถ์ ์ง์คํ  ์ ์๋ค. 
    3. `Data ๊ทธ๋ฃนํ` : ๊ด์ฌ ์๋ ๋ณต์๋ค(ํ์ ํ๋ณด ๊ฒฐ์ )์ ๋ฉ์ด๋ฆฌ๋ก ๋ฌถ๋๋ค. ๊ทธ๋ฆฌ๊ณ  ๊ฐ ํซ์คํ์ ๊ฐ๋ต์ ์ธ ์ค์ฌ๋ถ๋ฅผ ์ฐพ๋ ๊ณผ์ ์ ์งํํ๋ค. ๊ฒฐ์ ๋ง๋ค ์ธ๋ฑ์ค๊ฐ ๋ถ์ฌ๋๋ฉฐ ์ด ์ ๋ณด๋ ์ต์ข ๋ถ๋ฅ๊ธฐ์ ์ ๋ฌ๋๋ค. ์ด ๊ณผ์ ์ ํ์ดํ ์น๋ฅผ ์๊ตฌํ์ง ์์์ ๋ณ๋์ ๋จ๊ณ๋ก ๊ตฌ์ฑ๋์๋ค.
    4. `Classification(๋ถ๋ฅํ๊ธฐ)` : 3์ฐจ์ convolution์ ์ฌ์ฉํด ๊ฐ ํ๋ณด๊ฒฐ์ ์ ์ค์  ๊ฒฐ์ ์ธ์ง ์๋์ง ๊ฒฐ์ ํ๋ค. ์ด๋ ์๋ ฅ ๋ฐ์ดํฐ๋ฅผ ๊ฑธ๋ฌ๋ด๋ ๊ฒ๊ณผ, ๋๋ฌด ๊ฑธ๋ฌ๋ด๋ค๊ฐ ์ค์ํ ์ ๋ณด๊น์ง ๋ฒ๋ ค์ง๋ ๊ฒ ์ฌ์ด์์ ๊ท ํ์ ์ก์ ํ์๊ฐ ์๋ค. -> `overfitting ๊ณผ underfitting์ ๊ฐ๋์ผ๊น?`
    5. `Diagnose(์ง๋จ ๋ด๋ฆฌ๊ธฐ)` : ๊ฐ ๊ฒฐ์ ๋ณ๋ก ๋ถ๋ฅํ ๊ฒฐ๊ณผ๋ฅผ ์ขํฉํ์ฌ ํ์๋ฅผ ์ง๋จํ๋ค. ๊ฒฐ์ ๋ณ๋ก ์์ฑ ์์ธก์ ๊ตฌํ ํ ์ด๋ค์ ์ต๋๊ฐ์ ์ฌ์ฉํ๋ ๋ฐฉ์์ผ๋ก ํ๋๋ผ๋ ์์ฑ์ด ์์ธก๋ ๊ฒฝ์ฐ, ์ ์๊ฒฌ์ ๋ณด์ธ๋ค๊ณ  ๊ฒฐ์ ํ๋ค. 
- `3๋จ๊ณ`์ `4๋จ๊ณ`์์ ํ๋ จ์ ์ฌ์ฉํ๋ ๋ฐ์ดํฐ๋ ์ฌ๋์ด ํ์ธํ๊ณ  ํ๊ธฐํ ์ ๋ต์ ์ ๊ณตํ๋ค. -> `2๋จ๊ณ์ 3๋จ๊ณ, 4๋จ๊ณ๋ฅผ ์์์ ์๊ด์์ด ๊ฐ๋ณ๋ก ์งํํ  ์ ์๋ค`๊ฐ ๋ฌด์จ ์๋ฏธ์ผ๊น..?

### 9.4.3 ๋ฐ์ดํฐ ์์ค : LUNA ๊ทธ๋๋ ์ฑ๋ฆฐ์ง
- LUNA(LUng Nodule Analysis) 2016 ๊ทธ๋๋ ์ฑ๋ฆฐ์ง์์ ๊ฐ์ ธ์จ CT์ค์บ ๋ฐ์ดํฐ๋ฅผ ์ฌ์ฉํ๋ค.
    - ํ์์ CT์ค์บ ์๋ฃ๋ฅผ ๊ณ ํ์ง๋ก ๋ ์ด๋ธ ์์ํ ๊ณต๊ฐ ๋ฐ์ดํฐ์์.
- LUNA ๋ฐ์ดํฐ ๋ค์ด๋ก๋
    - ์์ถ์ ํ๋ฉด 120GB ๊ณต๊ฐ์ ์ฐจ์งํ๋ ๋ฐ์ดํฐ์.
    - CT๋ฐ์ดํฐ ์ ์ฒด๋ฅผ ์ฝ๋ ๋์  ํ์ํ ๋ถ๋ถ๋ง ์ฝ๋๋ก ๋ฐ์ดํฐ๋ฅผ ์์ ๋ฉ์ด๋ฆฌ๋ก ๋๋ ๋ 100GB ์ ๋์ ์บ์ ๊ณต๊ฐ์ด ํ์ํ๋ค.
    - ๋ฐ์ดํฐ๋ 10๊ฐ์ subset๋ค(subset0~subset9)๋ก ์ด๋ฃจ์ด์ง๊ณ , ๋ฐ์ดํฐ์ ํ์ผ ์์ถ์ ํ๋ฉด `code/data-unversioned/part2/luna/subset()`๊ณผ ๊ฐ์ ๊ฐ๋ณ ์๋ธ ๋๋ ํ ๋ฆฌ๊ฐ ๋ณด์ธ๋ค.
    - Window ์ฌ์ฉ์๋ 7-Zip(www.7-zip.org) ์น์ฌ์ดํธ์์ ์์ถ ํด์  ํ๋ก๊ทธ๋จ์ ๋ค์ด๋ฐ์ ์ ์๋ค.
    - `candidates.csv`์ `annotations.csv`ํ์ผ๋ ํ์ํ๋ค. ์ฑ์ ๊นํ๋ธ์ ์น์ฌ์ดํธ์ ์ฝ๋ ์์. -> code/data/part2/luna/*.csv
    - candidates ํ์ผ๊ณผ ํ ๊ฐ ์ด์์ ์๋ธ์ ํ์ผ์ ๋ด๋ ค๋ฐ์ ์์ถ์ ํผ ํ ์ ํํ ๊ฒฝ๋ก์ ๋ฃ์์ผ๋ฉด ๋ค์ ์ฅ์ ์์ ๊ฐ ์คํ๊ฐ๋ฅํ  ๊ฒ์ด๋ค. 



