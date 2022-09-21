# 10.1. 원본 CT 데이터 파일

CT 데이터는 메타데이터 헤더 정보가 포함된 .mhd 파일과 3차원 배열을 만들 원본 데이터 바이트를 포함하는 .raw 파일이 존재. 

CT 데이터에 적용할 좌표계 변환이 있어야 함 → LUNA에서 제공하는 애노테이션 데이터(각 결절의 좌표 목록, 악성 여부, 해당 CT 스캔의 시리즈 UID) 읽기 → 결절 좌표 좌표계 변환 → 결절의 중심에 해당하는 복셀의 인덱스, 행, 열 정보가 생김 → 후보 위치 (I, R, C) 좌표를 사용해 모델 입력으로 사용

모델 노이즈를 방지하기 위해 데이터 이상값 제거에 피처 엔지니어링 feature engineering 사용

# 10.2. LUNA 애노테이션 데이터 파싱

CSV 파일을 파싱하여 각 CT 스캔 중 관심 있는 부분 파악 → 좌표 정보, 해당 좌표 지점이 결절인지 여부, CT 스캔에 대한 고유 식별자 얻음

candidates.csv 파일에는 조직 덩어리가 결절일 가능성이 있는지와 악성 종양/양성 종양 여부 등의 정보가 들어 있음 → 전체 후보 리스트를 만드는 데 사용하여 후에 훈련/검증 데이터셋으로 나눔

annotations.csv 파일에는 결절로 플래그된 후보들에 대한 정보가 포함되어 있음 → 결절 크기의 분포로 가정하여 훈련/검증 데이터 만드는 데 사용

## 10.2.1. 훈련셋과 검증셋

표준 지도 학습 supervised learning 작업은 훈련셋 training set/검증셋 validation set으로 나뉨

## 10.2.2. 애노테이션 데이터와 후보 데이터 합치기

```python
from collections import namedtuple
#...27행
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

#CT 데이터가 빠져있으므로 훈련 샘플은 아님
#애노테이션 데이터를 다듬은 통합 인터페이스 -> 지저분한 데이터를 처리하는 영역
```

후보 정보 리스트는 결절의 상태, 결절의 직경, 순번, 중심점을 가짐.

```python
@functools.lru_cache(1) #표준 인메모리 캐싱 라이브러리
def getCandidateInfoList(requireOnDisk_bool=True):
		# requireOnDisk_bool : 디스크에 없는 데이터를 걸러내기 위해
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
```

데이터 파일을 파싱하는데 시간이 걸리므로 함수 호출 결과를 메모리에 캐시 → 데이터 파이프라인 속도를 올림

requireOnDisk_bool 파라미터를 이용하면 디스크상에서 시리즈 UID 가 발견되는 LUNA 데이터만 사용하고 이에 해당하는 엔트리만 CSV 파일에서 걸러 사용됨 → 코드 검증이 쉬어짐

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
# annotations.csv의 직경 정보를 합침
# 애노테이션 정보는 series_uid로 그룹화해서 두 파일에서 일치하는 행을 찾아내는 키로 사용

```

```python
candidateInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

# candidates.csv 정보를 이용해 전체 후보 리스트 만들기
# series_uid 에 해당하는 후보 엔트리에 대해 애노테이션 데이터를 루프 돌면서
# 같은 series_uid를 찾고 두 센터 좌표가 가까운지 확인
# 일치할 경우 결절에 대한 직경 정보를 매칭한 것
# 일치하지 않으면 결절의 직경이 0.0인 것으로 간주
# -> 훈련/검증셋에 대해 좋은 결절 크기 분포를 가지도록 필터링 함
```

```python
candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

# 데이터를 정렬-> 일부 CT 단면들을 모아 결절 직경에 대해 잘 분포된 실제 결절을 반영하는
# 덩어리를 얻을 수 있음
```

# 10.3. 개별 CT 스캔 로딩

디스크에서 CT 데이터 얻어와 파이썬 객체로 변환 → 3차원 결절 밀도 데이터로 사용

결절 애노테이션 정보는 원본 데이터에서 얻어내고자 하는 영역에 대한 맵 map에 해당 → 맵 이용해서 관심있는 부분 추출을 위해 먼저 데이터를 주소로 접근 가능하게 해야함

```python
import SimpleITK as sitk
class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
```

## 10.3.1. 하운스필드 단위 HU

CT 스캔 복셀은 하운스필드 단위로 표시 ex. 공기 -1000HU 물 0HU (환자 외부는 마이너스 값)

HU 값은 통상적으로 부호화된 12비트 정수로 디스크에 저장 (16비트 정수 공간 사용)

CT 스캐너가 제공하는 정밀도 레벨에 적당

```python
ct_a.clip(-1000, 1000, ct_a)

# 환자 몸 외부는 공기이므로 -1000HU 이하값음 모두 버림
# 뼈나 금속 이식물에 해당하는 밀도값도 필요없으므로 1000HU 이상인 경우도 버림
# -> outlier 제거하기
# 이렇게 만들어진 값을 self에 할당

self.series_uid = series_uid
self.hu_a = ct_a
```

# 10.4. 환자 좌표계를 사용해 결절 위치 정하기

입력 뉴런 수가 고정되어 있기 때문에 통상적인 딥러닝 모델은 고정된 크기의 입력을 필요로 함 →  분류기 입력으로 사용할 고정된 크기의 결절 후보를 담을 배열 필요 → 모델 훈련에 깔끔하게 잘린, 중심이 잘 잡힌 후보 데이터를 사용하도록 함

## 10.4.1. 환자 좌표계

10.2 에서 불러온 후보 중심점 데이터는 복셀이 아닌 밀리미터 기반 좌표계인 (X, Y, Z) → 복셀 주소 기반 좌표계인 (I 인덱스, R 행, C 열)로 좌표 변환

환자 좌표계는 밀리미터 단위로 측정하고, 각기 기준과 비율이 다름 → 해부학적으로 관심있는 위치를 지정하기 위해 사용

## 10.4.2. CT 스캔 형태와 복셀 크기

일반적으로 복셀은 정육면체가 아님. 대부분 행=열 이지만 인덱스는 수치가 조금 큼 → 데이터가 왜곡된 이미지처럼 보임 → 비율 계수 scale factor 적용해야함

## 10.4.3. 밀리미터를 복셀 주소로 변환하기

환자의 밀리미터 좌표와 (I, R, C) 배열 좌표 변환을 돕는 코드

복셀 인덱스를 좌표로 바꾸기 위한 순서

1. 좌표를 XYZ 체계로 만들기 위해 IRC 에서 CRI로 뒤집기
2. 인덱스를 복셀 크기로 확대축소
3. 파이썬의 @ 사용하여 방향을 나타내는 행렬과 행렬곱을 수행
4. 기준으로부터 오프셋 더하기

 

```python
IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] # 넘파이 배열로 변환하며 순서 바꾸기
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
		# 2, 3, 4 단계 실행
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
		# 4, 3, 2 단계 실행
    cri_a = np.round(cri_a) # 반올림
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0])) # 섞으면서 정수로 변환
```

```python
class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

# xtz2irc 변한 함수에 넘겨줄 개별 포인트 데이터와 함께 필요한 값
# 각 후보의 센터를 환자 좌표에서 배열좌표로 변환하기 위한 데이터 넣는 일 마무리
```

## 10.4.4. CT 스캔에서 결절 추출하기

각 후보 영역을 추출해 모델이 한 번에 한 영역에 집중할 수 있도록 만들기

```python
def getRawCandidate(self, center_xyz, width_irc):
# LUNA CSV 데이터에 명시된 환자 좌표계 (X, Y, Z) 로 표시된 중심 정보와 복셀 단위의
# 너비 정보도 인자로 받아 정육면체의 CT 덩어리와 배열 좌표로 변환된 후보의 중심값 반환
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc
```

# 10.5. 간단한 데이터셋 구현

LunaDataset 클래스로 샘플을 정규화하고 각 CT 결절은 평탄화하여 단일 컬렉션으로 합치기

```python
def __len__(self): # 초기화 후에 하나의 상수값 반환
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx): # 인덱스를 인자로 받아 훈련에서 사용할 샘플 데이터 튜플 반환
    # ndx 인자 (0~N-1)를 받아 네 개의 아이템이 있는 샘플 튜플로 반환    

		# ... 200행
		return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )
```

```python
def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

				candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0) # 채널 차원 추가

# 거친 데이터를 깔끔하고 정렬된 텐서로 바꿔주는 변환 작업

				pos_t = torch.tensor([
				                not candidateInfo_tup.isNodule_bool,
				                candidateInfo_tup.isNodule_bool
				            ],
				            dtype=torch.long,
				        ) # 분류 텐서
```

```python
# 최종 샘플 튜플 확인
# In [10]:
LunaDataset()[0]

#Out[10]:
(tensor([[[[-899., -903., ........],
				... ,
				[-92., -63., ...]]]]), # candidate_t
	tensor([0,1]), # cls_t
	'1.3.6.... ... ', # candidate_tup.series_uid
	tensor([91, 360, 341])) # center_irc
```

## 10.5.1. getCtRawCandidate 함수로 후보 배열 캐싱하기

LunaDataset으로부터 쓸만한 성능을 얻기 위해 온디스크 캐싱 on-disk caching → 전체 CT 스캔을 읽지 않아도 됨 → 프로젝트의 병목 지점을 파악하고 진행이 느려질 경우 최적화하는 방법 찾기

> 캐싱 : 컴퓨팅에서 캐시는 일반적으로 일시적인 특징이 있는 데이터 하위 집합을 저장하는 고속 데이터 스토리지 계층입니다. 따라서 이후에 해당 데이터에 대한 요청이 있을 경우 데이터의 기본 스토리지 위치에 액세스할 때보다 더 빠르게 요청을 처리할 수 있습니다. 캐싱을 사용하면 이전에 검색하거나 계산한 데이터를 효율적으로 재사용할 수 있습니다.
> 

```python
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)
# getCt 반환값을 메모리에 캐싱 -> 동일한 Ct 인스턴스에 대한 요청에 대해 엄청난 속도 향상

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc
# 출력 캐싱하지만 getCt는 아예 호출되지 않음 -> 디스크에 캐싱
```

## 10.5.2. LunaDataset.__init__으로 데이터셋 만들기

val_stride 파라미터 이용해 샘플 중 10번째에 해당하는 모든 경우를 검증셋으로 둠

isValSet_bool 파라미터로 훈련데이터나 검증데이터만 둘지 둘다 둘지를 지정

```python
class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
            ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid: # 코드에 series_uid 넣어주면 해당 데이터의 결절만 인스턴스에 담김
						# -> 문제가 있는 CT 스캔 하나만 자세히 볼 수 있음
						# -> 데이터 시각화나 디버깅에 유용
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
```

## 10.5.3. 훈련/검증 분리

Dataset의 N번째 데이터들만 따로 모아 모델 검증용 서브셋 만듦

```python
if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list 
						# self.candidateInfo_list에서 검증용 이미지 삭제하기
```

## 10.5.4. 데이터 렌더링
