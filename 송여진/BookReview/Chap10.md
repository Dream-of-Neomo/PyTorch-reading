# 10. 여러 데이터 소스를 통합 데이터셋으로 합치기

#### Goal 🏃  
원본 CT 스캔 데이터와 데이터에 달아놓은 애노테이션 `annotation` 목록으로 훈련 샘플 만들기  
> Data annotation ❓  
> 데이터 애노테이션은 데이터셋에 메타 데이터를 추가하는 작업을 말한다. 태그 형식으로 인공지능이 데이터의 내용을 알아들을 수 있게 주석을 달아주는 작업이라고 할 수 있다. 데이터 라벨링이라고 불리기도 한다.

## 1. 원본 CT 데이터 파일

CT 데이터는 메타데이터 헤더 정보가 포함된 `.mhd` 파일과 3차원 배열을 만들 원본 데이터 바이트를 포함하는 `.raw` 파일로 총 2가지 종류이다.  
`Ct` 클래스는 두 파일을 읽어서 3차원 배열을 만들고 환자 좌표계를 배열에서 필요로 하는 인덱스, 행, 열 좌표로 바꿔주는 변환 행렬도 만든다.  
또한 LUNA에서 제공하는 애노테이션 데이터도 읽어야 한다. 이 데이터에는 각 결절의 좌표 목록, 악성 여부, 그리고 해당 CT 스캔의 시리즈 UID가 포함되어 있다. 
`(I,R,C)` 좌표를 사용하면 CT 데이터의 작은 3차원 부분 단면을 얻어 모델 입력값으로 사용할 수 있다. 이 3차원 배열과 함께 우리는 튜플의 나머지를 구성해야 한다. 여기에는 샘플 배열, 결절의 상태 플래그, 시리즈 UID를 비롯해 결절 후보군의 CT 리스트 중 이 샘플이 몇 번째 인덱스인지 등이 포함된다. 파이토치가 `Dataset` 서브클래스를 통해 얻고자 하는 것이 이 튜플이다. 이 튜플은 또한 원본 데이터를 파이토치 텐서로 바꿔주는 과정의 마지막 부분이다.  

## 2. LUNA 애노테이션 데이터 파싱
> Data Parsing ❓  
> 데이터 파싱은 한국어로는 구문 분석, 정제하기 등으로 불리며 데이터가 알맞은 문법에 맞게 정리되었는지 확인하고, 파싱하는 데이터를 목적에 맞게 이용하기 쉬운 형태로 구성해주는 것을 말한다.  
  
`candidates.csv` 파일을 파싱한다. 이 파일에는 조직 덩어리가 결절일 가능성이 있는지와 악성 종양 또는 양성 종양 여부, 그리고 그 밖의 정보가 들어 있다. 우리는 이 데이터를 추후에 훈련 dataset과 검증 dataset으로 나눌 것이다.  
```console
(base) song-yeojin@song-yeojin-ui-MacBookAir code % wc -l candidates.csv
  551066 candidates.csv
(base) song-yeojin@song-yeojin-ui-MacBookAir code % head candidates.csv
seriesuid,coordX,coordY,coordZ,class
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-56.08,-67.85,-311.92,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,53.21,-244.41,-245.17,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,103.66,-121.8,-286.62,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-33.66,-72.75,-308.41,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-32.25,-85.36,-362.51,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-26.65,-203.07,-165.07,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-74.99,-114.79,-311.92,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-16.14,-248.61,-239.55,0
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,135.89,-141.41,-252.2,0
(base) song-yeojin@song-yeojin-ui-MacBookAir code % grep ",1$" candidates.csv | wc -l
       0
 ```
> 마지막 명령어가 잘 작동하지 않아서 Mac의 문제인가 싶어서 Google Colab에서도 해보았는데 여전히 안된다..아무래도 ,이나 $ 쪽에서 문제가 생긴 것 같은데 ㅠㅠ  
  
  코드를 보면 wc -l은 파일의 행 개수를 카운트한다. 즉 `candidates.csv`에는 총 551066개의 행이 있음을 알 수 있다. head 명령어로 파일의 앞부분 일부를 출력하고, 결절은 1로 나타나므로 1로 끝나는 행의 개수를 센다. 
 
 `annotations.csv` 파일에는 결절로 플래그된 후보들에 대한 정보가 포함되어 있다. 이 중 `diameter_mm` 정보를 주목할 만 하다. 
 ```console
 (base) song-yeojin@song-yeojin-ui-MacBookAir code % wc -l annotations.csv
    1187 annotations.csv
(base) song-yeojin@song-yeojin-ui-MacBookAir code % head annotations.csv
seriesuid,coordX,coordY,coordZ,diameter_mm
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-128.6994211,-175.3192718,-298.3875064,5.651470635
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,103.7836509,-211.9251487,-227.12125,4.224708481
1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208,69.63901724,-140.9445859,876.3744957,5.786347814
1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405,-24.0138242,192.1024053,-391.0812764,8.143261683
1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405,2.441546798,172.4648812,-405.4937318,18.54514997
1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405,90.93171321,149.0272657,-426.5447146,18.20857028
1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405,89.54076865,196.4051593,-515.0733216,16.38127631
1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275,81.50964574,54.9572186,-150.3464233,10.36232088
1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249,105.0557924,19.82526014,-91.24725078,21.08961863
```   
wc -l 명령어를 사용했을 때 나오는 값이 `candidates.csv`와 다르다. 또한 마지막의 열 정보 또한 다르다. 
   
   
### 2.1. 훈련셋과 검증셋   
모든 표준 지도 학습 `supervised learning` 작업은 데이터를 훈련셋 `training set`과 검증셋 `validation set`으로 나눈다.  
먼저 데이터를 크기 순으로 정렬한 후 매 N번째에 대해 검증셋에 넣어서 분포를 반영한 검증셋을 구성한다. 하지만 `annotations.csv`에서 제공하는 위치 정보는 `candidates.csv`의 좌표와 정확하게 일치하지는 않는다.  
```console
(base) song-yeojin@song-yeojin-ui-MacBookAir code % grep 100225287222365663678666836860 annotations.csv
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,-128.6994211,-175.3192718,-298.3875064,5.651470635
1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860,103.7836509,-211.9251487,-227.12125,4.224708481
(base) song-yeojin@song-yeojin-ui-MacBookAir code % grep '100225287222365663678666836860.*,1$' candidates.csv
```  
> 또 작동이 안된다..진짜뮈치겠따.....   

각 파일에서 일치하는 좌표를 잘라내면 **(-128.70, -175.32, -298.39)** 와 **(-128.94, -175.04, -297.87)** 을 얻는다. 이 결절은 직경이 5mm이며 두 좌표는 정확하게 결절의 중심부를 나타내지만 두 좌표가 완벽히 일치하지는 않는다. 이와 같은 데이터는 가치가 없는 것으로 판단되어 무시한다.
  
### 2.2. 애노테이션 데이터와 후보 데이터 합치기

이제 이 두 데이터를 합치는 `GetCandidateInfoList` 함수를 만든다. 각 결절 정보를 담아둘 네임드 튜플 `named tuple` 을 파일 상단에 두고 사용한다. 
```python
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)
```   
`dsets.py` 파일의 27번째 행이다. 이 튜플은 우리가 원하는 CT 데이터가 빠져 있으므로 훈련 샘플이 아니다. 
후보 정보 리스트는 결절의 상태, 결절의 직경, 순번과 중심점을 갖는다. `NoduleInfoTuple` 인스턴스 리스트를 만드는 함수는 인메모리 캐싱 데코레이터 `in-memory caching decorator`를 사용하고 디스크 파일 경로를 얻는다.
```python
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
```
`requireOnDisk_bool`은 디스크에 없는 데이터는 걸러내기 위함이다. 일부 데이터 파일은 파싱에 시간이 걸리므로 함수 호출 결과를 메모리에 캐시한다.  
앞서 훈련 데이터셋 전체를 사용하면 다운로드도 오래 걸리고 필요한 디스크 공간도 커지기 때문에 훈련 프로그램에 집중하겠다고 말한 바가 있는데, `requireOnDisk_bool`을 사용하여 LUNA 데이터만 사용할 수 있다.   
적절한 후보 정보를 얻었다면 `annotations.csv` 의 직경 정보를 합치자. 애노테이션 정보는 `series_uid` 로 그룹화하여 두 파일에서 일치하는 행을 찾아내는 키로 사용한다.
```python
diameter_dict = {}
    with open('data/part2/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
 ```  
 그리고 이제 `candidates.csv` 의 정보를 사용하여 전체 후보 리스트를 만든다.   
 ```python
    candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool: #series_uid가 없으면 서브셋엔 있지만 디스크엔 없어서 건너뜀
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4: # 반경을 얻기 위해 직경을 2로 나누고 결절 센터가 크기 기준으로 떨어진 정도를 반지름의 절반 길이를 기준으로 판정
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))
 ```   
 주어진 `series_uid`에 해당하는 후보 엔트리에 대해 앞에서 모은 애노테이션 데이터를 루프 돌리면서 같은 `series_uid`를 찾고 두 센터의 좌표가 같은 결절로 간주할 만큼 가까운 거리에 있는지를 확인한다. 일치하지 않는다면 직경이 0.0인 것으로 간주한다.  
 이제 데이터를 정렬 후 반환한다.
 ```python
     candidateInfo_list.sort(reverse=True) # 내림차순 정렬
    return candidateInfo_list
  ```   
  `noduleInfo_list`의 튜플 멤버 순서는 이 정렬로 만들어진다. 이렇게 데이터를 정렬하면 일부 CT 단면들을 모아 결절 직경에 대해 잘 분포된 실제 결절을 반영하는 덩어리를 얻어올 수 있게 된다.   
  ## 3. 개별 CT 스캔 로딩
  다음은 디스크에서 CT 데이터를 얻어와 파이썬 객체로 변환해서 3차원 결절 밀도 데이터로 사용할 수 있도록 만드는 작업이다. 결절 애노테이션 정보는 원본 데이터에서 얻어내고자 하는 영역에 대한 맵이라고 생각하면 된다. 이 맵을 활용하여 관심 있는 부분을 추출하려면 데이터를 주소로 접근 가능하게 만들어야 한다.  
  CT 스캔 파일의 원래 포맷은 `DICOM`이라고 부른다. 이 시절에 만들어진 내용들은 다소 깔끔하지 않다. 하지만 `LUNA`에서는 데이터를 `MetaIO` 포맷으로 변환해 놓았고 사용하기에도 쉽다. :point_right: [링크](https://itk.org/Wiki/MetaIO/Documentation#Quick_Start)  
  데이터 파일 포맷을 블랙박스로 간주하고 친숙한 numpy 배열로 읽어들이기 위해 `SimpleITK`를 사용할 것이다. 
  ```python
  class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid) #서브셋 위치 상관없으므로 와일드카드 사용
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path) #sitk는 .raw 파일도 읽음
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32) #타입을 np.float3으로 변환하기 위해 array 다시만듬
```  
주어진 데이터 샘플을 식별하기 위해 우리는 시리즈 인스턴스 UID `series_uid`를 이용하고 있다. `DICOM`은 개별 파일, 파일 그룹, 처리 과정 등에서 단일 식별자 `UID`를 주로 사용하고 있다. 식별자는 [`UUID`](https://docs.python.org/3.6/library/uuid.html) 과 흡사하지만 생성하는 방식과 포맷은 다르다. 여기에서는 UID를 여러 CT 스캔을 참조 시 사용할 ASCII 문자열로 만든 단일 키로 취급한다.  
우리의 데이터 중 10개의 서브셋에는 각각 90개의 CT 스캔이 있으며 각 CT 스캔은 `.mhd`와 `.raw` 확장자를 가지는 두 가지 파일로 나뉜다. `ct_a` 는 3차원 배열이다. 세 개의 차원은 공간을 나타내고 하나는 밀도를 나타낸다. 파이토치 텐서에서 채널 정보는 네 번째 차원으로 표현되며 크기는 1이다.  
  
****
### 3.1. 하운스필드 단위(Hounsfield Unit, HU)  
`__init__` 메소드 작성 시에는 `ct_a` 값을 지워줄 필요가 있다. CT 스캔 복셀은 [하운스필드 단위](https://en.wikipedia.org/wiki/Houndfield_scale)로 표시되어 있는데, 예시로 공기는 -1,000HU이고 물은 0HU이며 뼈는 +1,000HU이다. HU 값은 통상적으로 부호화된 12비트 정수로 디스크에 저장한다.  
어떤 CT 스캐너는 스캔 영역을 벗어난 복셀을 나타내기 위해 밀도에 음의 값을 사용한다. 우리의 경우 환자의 몸 외부는 모두 공기이므로 값이 -1,000HU 이하인 경우에는 값을 버린다. 또한 뼈나 금속 이식물에 해당하는 밀도값도 필요 없으므로 2g/cc(1,000HU) 이상인 경우에도 잘라낸다. 
```python
 ct_a.clip(-1000, 1000, ct_a)
 ```  
 우리가 관심있는 종양의 경우 대체로 1g/cc(0HU) 근처이다. 따라서 1g/cc가 아닌 경우에도 걸러낸다. 데이터에서 이와 같이 이상값 `outlier`을 제거한다. 이렇게 만들어진 값을 self에 할당한다.
 ```pyton
 self.series_uid = series_uid
 self.hu_a = ct_a
 ```   

## 4. 환자 좌표계를 사용해 결절 위치 정하기   
통상적으로 딥러닝 모델은 고정된 크기의 입력을 필요로 한다. 입력 뉴런 수가 고정되어 있기 때문이다. 모델의 훈련에는 CT 스캔에서 깔끔하게 잘라낸 중심이 잘 잡힌 후보 데이터를 사용해서 모델이 입력 언저리에 감춰진 결절을 탐지해내는 일은 없게 할 예정이다.  

### 4.1. 환자 좌표계   
우리가 전에 읽어들인 후보 중심점 데이터는 복셀이 아니라 밀리미터 단위로 표시되어 있다. 따라서 좌표를 밀리미터 기반 좌표계인 `(X,Y,Z)` 에서 복셀 주소 기반 좌표계인 `(I,R,C)`로 변환해야 한다.   
환자 좌표계에서 X는 환자의 왼쪽이며, 양의 Y값은 환자의 뒤쪽(후면)이고, 양의 Z값은 환자의 머리 방향(상부)이다. 왼쪽-후면-상부 `left-posterior-superior`를 줄여 `LPS`라고도 한다.CT 배열과 환자 좌표계 사이의 관계를 정의하는 메타데이터는 DICOM 파일의 헤더에 저장되어 있고 헤더 내에서 메타 영상 형식으로 지정되어 있다. 이 메타데이터는 `(X,Y,Z)` 에서 `(I,R,C)` 으로의 변환을 가능하게 한다.  

### 4.2. CT 스캔 형태와 복셀 크기 
CT 스캔마다 조금씩 다른 부분 중 하나는 복셀의 크기이다. 일반적으로 복셀은 정육면체가 아니다. 행과 열은 수치가 같고, 인덱스는 수치가 조금 크다. 따라서 복셀은 데이터를 정방형의 픽셀로 그려내면 왜곡된 이미지를 보이게 된다. 따라서 실제 비율로 보려면 비율 계수`scale factor`를 적용해야 한다.  
CT는 일반적으로 512행, 512열로 구성되며 인덱스 차원은 총 대략 100개에서 200개의 단면으로 이루어진다. 이는 약 2^25승개의 복셀에 해당하며 3,200만개의 데이터 포인트이다. 각 CT는 파일 메타데이터 내에 복셀의 크기를 밀리미터 단위로 정의하며, 이를 참조하기 위해 `ct_mhd.GetSpacing()`을 호출하고 있다. 

### 4.3. 밀리미터를 복셀 주소로 변환하기  
환자의 밀리미터 좌표(코드에서 `_xyz`로 끝나는 이름을 사용한다.) 와 (I,R,C) 배열 좌표 (코드에서는 `_irc` 로 끝나는 이름을 사용한다.) 변환을 돕기 위한 유틸리티 코드를 정의하자.  
`SimpleITK` 라이브러리에서 
