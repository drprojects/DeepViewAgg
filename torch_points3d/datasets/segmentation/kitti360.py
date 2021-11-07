import os
import os.path as osp
import numpy as np
import torch
from plyfile import PlyData
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Sampler
import logging
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm as tq
from collections import namedtuple

import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.datasets.segmentation import IGNORE_LABEL as IGNORE
from torch_points3d.metrics.kitti360_tracker import KITTI360Tracker

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

########################################################################
#                              Data splits                             #
########################################################################

WINDOWS = {
    'train': [
        '2013_05_28_drive_0000_sync/004631_004927',
        '2013_05_28_drive_0000_sync/000372_000610',
        '2013_05_28_drive_0000_sync/009886_010098',
        '2013_05_28_drive_0000_sync/007438_007605',
        '2013_05_28_drive_0000_sync/007044_007286',
        '2013_05_28_drive_0000_sync/000834_001286',
        '2013_05_28_drive_0000_sync/002501_002706',
        '2013_05_28_drive_0000_sync/010352_010588',
        '2013_05_28_drive_0000_sync/004916_005264',
        '2013_05_28_drive_0000_sync/001537_001755',
        '2013_05_28_drive_0000_sync/007277_007447',
        '2013_05_28_drive_0000_sync/003711_003928',
        '2013_05_28_drive_0000_sync/004093_004408',
        '2013_05_28_drive_0000_sync/003463_003724',
        '2013_05_28_drive_0000_sync/010830_011124',
        '2013_05_28_drive_0000_sync/008496_008790',
        '2013_05_28_drive_0000_sync/000599_000846',
        '2013_05_28_drive_0000_sync/001740_001991',
        '2013_05_28_drive_0000_sync/002913_003233',
        '2013_05_28_drive_0000_sync/002282_002514',
        '2013_05_28_drive_0000_sync/003221_003475',
        '2013_05_28_drive_0000_sync/002695_002925',
        '2013_05_28_drive_0000_sync/008278_008507',
        '2013_05_28_drive_0000_sync/005880_006165',
        '2013_05_28_drive_0000_sync/009003_009677',
        '2013_05_28_drive_0000_sync/009666_009895',
        '2013_05_28_drive_0000_sync/006387_006634',
        '2013_05_28_drive_0000_sync/006623_006851',
        '2013_05_28_drive_0000_sync/001980_002295',
        '2013_05_28_drive_0000_sync/007968_008291',
        '2013_05_28_drive_0000_sync/006828_007055',
        '2013_05_28_drive_0000_sync/010078_010362',
        '2013_05_28_drive_0000_sync/010577_010841',
        '2013_05_28_drive_0000_sync/006154_006400',
        '2013_05_28_drive_0000_sync/005249_005900',
        '2013_05_28_drive_0000_sync/011079_011287',
        '2013_05_28_drive_0000_sync/011278_011467',
        '2013_05_28_drive_0000_sync/000002_000385',
        '2013_05_28_drive_0000_sync/007596_007791',
        '2013_05_28_drive_0000_sync/008779_009015',
        '2013_05_28_drive_0000_sync/001270_001549',
        '2013_05_28_drive_0000_sync/004397_004645',
        '2013_05_28_drive_0000_sync/007777_007982',
        '2013_05_28_drive_0000_sync/003919_004105',
        '2013_05_28_drive_0002_sync/009885_010251',
        '2013_05_28_drive_0002_sync/014337_014499',
        '2013_05_28_drive_0002_sync/009049_009275',
        '2013_05_28_drive_0002_sync/006383_006769',
        '2013_05_28_drive_0002_sync/012197_012403',
        '2013_05_28_drive_0002_sync/008091_008324',
        '2013_05_28_drive_0002_sync/012378_012617',
        '2013_05_28_drive_0002_sync/008311_008656',
        '2013_05_28_drive_0002_sync/009502_009899',
        '2013_05_28_drive_0002_sync/007489_007710',
        '2013_05_28_drive_0002_sync/008645_009059',
        '2013_05_28_drive_0002_sync/007700_007935',
        '2013_05_28_drive_0002_sync/007925_008100',
        '2013_05_28_drive_0002_sync/015189_015407',
        '2013_05_28_drive_0002_sync/013652_013860',
        '2013_05_28_drive_0002_sync/011675_011894',
        '2013_05_28_drive_0002_sync/010819_011089',
        '2013_05_28_drive_0002_sync/014848_015027',
        '2013_05_28_drive_0002_sync/004613_004846',
        '2013_05_28_drive_0002_sync/010484_010836',
        '2013_05_28_drive_0002_sync/011885_012047',
        '2013_05_28_drive_0002_sync/013850_014120',
        '2013_05_28_drive_0002_sync/012607_012785',
        '2013_05_28_drive_0002_sync/011467_011684',
        '2013_05_28_drive_0002_sync/007002_007228',
        '2013_05_28_drive_0002_sync/005506_005858',
        '2013_05_28_drive_0002_sync/014106_014347',
        '2013_05_28_drive_0002_sync/009265_009515',
        '2013_05_28_drive_0002_sync/005847_006086',
        '2013_05_28_drive_0002_sync/007216_007502',
        '2013_05_28_drive_0002_sync/011082_011480',
        '2013_05_28_drive_0002_sync/004835_005136',
        '2013_05_28_drive_0002_sync/005317_005517',
        '2013_05_28_drive_0002_sync/014491_014687',
        '2013_05_28_drive_0002_sync/015540_015692',
        '2013_05_28_drive_0002_sync/015017_015199',
        '2013_05_28_drive_0002_sync/014677_014858',
        '2013_05_28_drive_0002_sync/006757_007020',
        '2013_05_28_drive_0002_sync/013409_013661',
        '2013_05_28_drive_0002_sync/006069_006398',
        '2013_05_28_drive_0002_sync/010237_010495',
        '2013_05_28_drive_0002_sync/015399_015548',
        '2013_05_28_drive_0002_sync/012776_013003',
        '2013_05_28_drive_0002_sync/004391_004625',
        '2013_05_28_drive_0002_sync/012988_013420',
        '2013_05_28_drive_0002_sync/015684_015885',
        '2013_05_28_drive_0002_sync/005125_005328',
        '2013_05_28_drive_0002_sync/012039_012206',
        '2013_05_28_drive_0003_sync/000617_000738',
        '2013_05_28_drive_0003_sync/000002_000282',
        '2013_05_28_drive_0003_sync/000508_000623',
        '2013_05_28_drive_0003_sync/000886_001009',
        '2013_05_28_drive_0003_sync/000394_000514',
        '2013_05_28_drive_0003_sync/000274_000401',
        '2013_05_28_drive_0003_sync/000731_000893',
        '2013_05_28_drive_0004_sync/007610_007773',
        '2013_05_28_drive_0004_sync/010785_011115',
        '2013_05_28_drive_0004_sync/002897_003118',
        '2013_05_28_drive_0004_sync/010010_010166',
        '2013_05_28_drive_0004_sync/006637_006868',
        '2013_05_28_drive_0004_sync/010327_010554',
        '2013_05_28_drive_0004_sync/008103_008330',
        '2013_05_28_drive_0004_sync/005466_005775',
        '2013_05_28_drive_0004_sync/003570_003975',
        '2013_05_28_drive_0004_sync/008547_008806',
        '2013_05_28_drive_0004_sync/007232_007463',
        '2013_05_28_drive_0004_sync/006306_006457',
        '2013_05_28_drive_0004_sync/003356_003586',
        '2013_05_28_drive_0004_sync/005157_005564',
        '2013_05_28_drive_0004_sync/007919_008113',
        '2013_05_28_drive_0004_sync/005765_005945',
        '2013_05_28_drive_0004_sync/004174_004380',
        '2013_05_28_drive_0004_sync/007045_007242',
        '2013_05_28_drive_0004_sync/010544_010799',
        '2013_05_28_drive_0004_sync/008320_008559',
        '2013_05_28_drive_0004_sync/004919_005171',
        '2013_05_28_drive_0004_sync/005930_006119',
        '2013_05_28_drive_0004_sync/009458_009686',
        '2013_05_28_drive_0004_sync/003107_003367',
        '2013_05_28_drive_0004_sync/008794_009042',
        '2013_05_28_drive_0004_sync/011105_011325',
        '2013_05_28_drive_0004_sync/007763_007929',
        '2013_05_28_drive_0004_sync/007449_007619',
        '2013_05_28_drive_0004_sync/009244_009469',
        '2013_05_28_drive_0004_sync/010156_010336',
        '2013_05_28_drive_0004_sync/006111_006313',
        '2013_05_28_drive_0004_sync/006857_007055',
        '2013_05_28_drive_0004_sync/003967_004185',
        '2013_05_28_drive_0004_sync/004708_004929',
        '2013_05_28_drive_0004_sync/009675_010020',
        '2013_05_28_drive_0004_sync/009026_009253',
        '2013_05_28_drive_0004_sync/004370_004726',
        '2013_05_28_drive_0004_sync/006450_006647',
        '2013_05_28_drive_0005_sync/005579_005788',
        '2013_05_28_drive_0005_sync/003245_003509',
        '2013_05_28_drive_0005_sync/002447_002823',
        '2013_05_28_drive_0005_sync/001386_001669',
        '2013_05_28_drive_0005_sync/002807_003311',
        '2013_05_28_drive_0005_sync/004549_004787',
        '2013_05_28_drive_0005_sync/004998_005335',
        '2013_05_28_drive_0005_sync/001653_001877',
        '2013_05_28_drive_0005_sync/004007_004299',
        '2013_05_28_drive_0005_sync/003501_003711',
        '2013_05_28_drive_0005_sync/000002_000357',
        '2013_05_28_drive_0005_sync/001189_001398',
        '2013_05_28_drive_0005_sync/001865_002132',
        '2013_05_28_drive_0005_sync/005777_006097',
        '2013_05_28_drive_0005_sync/000579_000958',
        '2013_05_28_drive_0005_sync/004771_005011',
        '2013_05_28_drive_0005_sync/000341_000592',
        '2013_05_28_drive_0005_sync/006298_006541',
        '2013_05_28_drive_0005_sync/004277_004566',
        '2013_05_28_drive_0005_sync/003698_004017',
        '2013_05_28_drive_0005_sync/002115_002461',
        '2013_05_28_drive_0005_sync/005324_005591',
        '2013_05_28_drive_0005_sync/000864_001199',
        '2013_05_28_drive_0005_sync/006086_006307',
        '2013_05_28_drive_0006_sync/004368_004735',
        '2013_05_28_drive_0006_sync/001208_001438',
        '2013_05_28_drive_0006_sync/007228_007465',
        '2013_05_28_drive_0006_sync/007457_007651',
        '2013_05_28_drive_0006_sync/004920_005128',
        '2013_05_28_drive_0006_sync/008271_008499',
        '2013_05_28_drive_0006_sync/001906_002133',
        '2013_05_28_drive_0006_sync/006393_006648',
        '2013_05_28_drive_0006_sync/007641_007836',
        '2013_05_28_drive_0006_sync/008898_009046',
        '2013_05_28_drive_0006_sync/004723_004930',
        '2013_05_28_drive_0006_sync/002511_002810',
        '2013_05_28_drive_0006_sync/000002_000403',
        '2013_05_28_drive_0006_sync/002124_002289',
        '2013_05_28_drive_0006_sync/001423_001711',
        '2013_05_28_drive_0006_sync/008490_008705',
        '2013_05_28_drive_0006_sync/005303_005811',
        '2013_05_28_drive_0006_sync/005107_005311',
        '2013_05_28_drive_0006_sync/006639_006827',
        '2013_05_28_drive_0006_sync/008694_008906',
        '2013_05_28_drive_0006_sync/004058_004393',
        '2013_05_28_drive_0006_sync/003001_003265',
        '2013_05_28_drive_0006_sync/005957_006191',
        '2013_05_28_drive_0006_sync/003895_004070',
        '2013_05_28_drive_0006_sync/003251_003634',
        '2013_05_28_drive_0006_sync/006177_006404',
        '2013_05_28_drive_0006_sync/000754_001010',
        '2013_05_28_drive_0006_sync/007027_007239',
        '2013_05_28_drive_0006_sync/009383_009570',
        '2013_05_28_drive_0006_sync/000387_000772',
        '2013_05_28_drive_0006_sync/001700_001916',
        '2013_05_28_drive_0006_sync/009038_009223',
        '2013_05_28_drive_0006_sync/008052_008284',
        '2013_05_28_drive_0006_sync/005801_005966',
        '2013_05_28_drive_0006_sync/007826_008063',
        '2013_05_28_drive_0006_sync/006818_007040',
        '2013_05_28_drive_0006_sync/001000_001219',
        '2013_05_28_drive_0006_sync/003613_003905',
        '2013_05_28_drive_0006_sync/009213_009393',
        '2013_05_28_drive_0006_sync/002801_003011',
        '2013_05_28_drive_0006_sync/002280_002615',
        '2013_05_28_drive_0007_sync/000002_000125',
        '2013_05_28_drive_0007_sync/002782_002902',
        '2013_05_28_drive_0007_sync/000293_000383',
        '2013_05_28_drive_0007_sync/001122_001227',
        '2013_05_28_drive_0007_sync/000208_000298',
        '2013_05_28_drive_0007_sync/001841_001957',
        '2013_05_28_drive_0007_sync/000542_000629',
        '2013_05_28_drive_0007_sync/000119_000213',
        '2013_05_28_drive_0007_sync/000624_000710',
        '2013_05_28_drive_0007_sync/001659_001750',
        '2013_05_28_drive_0007_sync/001577_001664',
        '2013_05_28_drive_0007_sync/002237_002410',
        '2013_05_28_drive_0007_sync/000705_000790',
        '2013_05_28_drive_0007_sync/001221_001348',
        '2013_05_28_drive_0007_sync/002395_002789',
        '2013_05_28_drive_0007_sync/000785_000870',
        '2013_05_28_drive_0007_sync/001034_001127',
        '2013_05_28_drive_0007_sync/000378_000466',
        '2013_05_28_drive_0007_sync/000947_001039',
        '2013_05_28_drive_0007_sync/001483_001582',
        '2013_05_28_drive_0007_sync/000865_000952',
        '2013_05_28_drive_0007_sync/001950_002251',
        '2013_05_28_drive_0007_sync/000461_000547',
        '2013_05_28_drive_0007_sync/001745_001847',
        '2013_05_28_drive_0007_sync/001340_001490',
        '2013_05_28_drive_0009_sync/011351_011646',
        '2013_05_28_drive_0009_sync/005422_005732',
        '2013_05_28_drive_0009_sync/008953_009208',
        '2013_05_28_drive_0009_sync/001234_001393',
        '2013_05_28_drive_0009_sync/012683_012899',
        '2013_05_28_drive_0009_sync/003712_003987',
        '2013_05_28_drive_0009_sync/001385_001543',
        '2013_05_28_drive_0009_sync/009195_009502',
        '2013_05_28_drive_0009_sync/002826_003034',
        '2013_05_28_drive_0009_sync/007264_007537',
        '2013_05_28_drive_0009_sync/006272_006526',
        '2013_05_28_drive_0009_sync/007838_008107',
        '2013_05_28_drive_0009_sync/003188_003457',
        '2013_05_28_drive_0009_sync/000451_000633',
        '2013_05_28_drive_0009_sync/000284_000460',
        '2013_05_28_drive_0009_sync/010703_011118',
        '2013_05_28_drive_0009_sync/013133_013380',
        '2013_05_28_drive_0009_sync/001005_001244',
        '2013_05_28_drive_0009_sync/003441_003725',
        '2013_05_28_drive_0009_sync/012167_012410',
        '2013_05_28_drive_0009_sync/000623_000787',
        '2013_05_28_drive_0009_sync/005719_005993',
        '2013_05_28_drive_0009_sync/013575_013709',
        '2013_05_28_drive_0009_sync/001534_001694',
        '2013_05_28_drive_0009_sync/001951_002126',
        '2013_05_28_drive_0009_sync/007038_007278',
        '2013_05_28_drive_0009_sync/000778_001026',
        '2013_05_28_drive_0009_sync/000002_000292',
        '2013_05_28_drive_0009_sync/005156_005440',
        '2013_05_28_drive_0009_sync/002117_002353',
        '2013_05_28_drive_0009_sync/010086_010717',
        '2013_05_28_drive_0009_sync/002615_002835',
        '2013_05_28_drive_0009_sync/009727_010097',
        '2013_05_28_drive_0009_sync/008391_008694',
        '2013_05_28_drive_0009_sync/001686_001961',
        '2013_05_28_drive_0009_sync/004905_005179',
        '2013_05_28_drive_0009_sync/008096_008413',
        '2013_05_28_drive_0009_sync/009489_009738',
        '2013_05_28_drive_0009_sync/004475_004916',
        '2013_05_28_drive_0009_sync/013701_013838',
        '2013_05_28_drive_0009_sync/003026_003200',
        '2013_05_28_drive_0009_sync/003972_004258',
        '2013_05_28_drive_0009_sync/005976_006285',
        '2013_05_28_drive_0009_sync/012398_012693',
        '2013_05_28_drive_0009_sync/011896_012181',
        '2013_05_28_drive_0009_sync/007524_007859',
        '2013_05_28_drive_0009_sync/006740_007052',
        '2013_05_28_drive_0009_sync/012876_013148',
        '2013_05_28_drive_0009_sync/002342_002630',
        '2013_05_28_drive_0009_sync/011630_011912',
        '2013_05_28_drive_0009_sync/013370_013582',
        '2013_05_28_drive_0009_sync/011099_011363',
        '2013_05_28_drive_0009_sync/004246_004489',
        '2013_05_28_drive_0009_sync/008681_008963',
        '2013_05_28_drive_0009_sync/006515_006753',
        '2013_05_28_drive_0010_sync/000199_000361',
        '2013_05_28_drive_0010_sync/002911_003114',
        '2013_05_28_drive_0010_sync/002024_002177',
        '2013_05_28_drive_0010_sync/001563_001733',
        '2013_05_28_drive_0010_sync/002756_002920',
        '2013_05_28_drive_0010_sync/003106_003313',
        '2013_05_28_drive_0010_sync/001245_001578',
        '2013_05_28_drive_0010_sync/001872_002033',
        '2013_05_28_drive_0010_sync/000002_000208',
        '2013_05_28_drive_0010_sync/002168_002765',
        '2013_05_28_drive_0010_sync/000854_000991',
        '2013_05_28_drive_0010_sync/000718_000881',
        '2013_05_28_drive_0010_sync/000549_000726',
        '2013_05_28_drive_0010_sync/001109_001252',
        '2013_05_28_drive_0010_sync/001724_001879',
        '2013_05_28_drive_0010_sync/000984_001116',
        '2013_05_28_drive_0010_sync/000353_000557'],

    # These are 3 randomly-picked windows also in train for now. Need
    # to define a true validation set with all classes represented and
    # with no overlap with train.
    'val': [
        '2013_05_28_drive_0000_sync/009886_010098',
        '2013_05_28_drive_0009_sync/001005_001244',
        '2013_05_28_drive_0010_sync/000549_000726'],

    'test': [
        '2013_05_28_drive_0008_sync/0000006988_0000007177',
        '2013_05_28_drive_0008_sync/0000000002_0000000245',
        '2013_05_28_drive_0008_sync/0000008536_0000008643',
        '2013_05_28_drive_0008_sync/0000000235_0000000608',
        '2013_05_28_drive_0008_sync/0000008417_0000008542',
        '2013_05_28_drive_0008_sync/0000004623_0000004876',
        '2013_05_28_drive_0008_sync/0000001277_0000001491',
        '2013_05_28_drive_0008_sync/0000004854_0000005104',
        '2013_05_28_drive_0008_sync/0000006792_0000006997',
        '2013_05_28_drive_0008_sync/0000002769_0000003002',
        '2013_05_28_drive_0008_sync/0000006247_0000006553',
        '2013_05_28_drive_0008_sync/0000007875_0000008100',
        '2013_05_28_drive_0008_sync/0000000812_0000001058',
        '2013_05_28_drive_0008_sync/0000007161_0000007890',
        '2013_05_28_drive_0008_sync/0000008236_0000008426',
        '2013_05_28_drive_0008_sync/0000001046_0000001295',
        '2013_05_28_drive_0008_sync/0000006517_0000006804',
        '2013_05_28_drive_0008_sync/0000005911_0000006258',
        '2013_05_28_drive_0008_sync/0000008637_0000008745',
        '2013_05_28_drive_0008_sync/0000005316_0000005605',
        '2013_05_28_drive_0008_sync/0000008090_0000008242',
        '2013_05_28_drive_0008_sync/0000005588_0000005932',
        '2013_05_28_drive_0008_sync/0000002580_0000002789',
        '2013_05_28_drive_0008_sync/0000005093_0000005329',
        '2013_05_28_drive_0008_sync/0000000581_0000000823',
        '2013_05_28_drive_0008_sync/0000002404_0000002590',
        '2013_05_28_drive_0018_sync/0000001191_0000001409',
        '2013_05_28_drive_0018_sync/0000001399_0000001587',
        '2013_05_28_drive_0018_sync/0000003503_0000003724',
        '2013_05_28_drive_0018_sync/0000002090_0000002279',
        '2013_05_28_drive_0018_sync/0000002487_0000002835',
        '2013_05_28_drive_0018_sync/0000002827_0000003047',
        '2013_05_28_drive_0018_sync/0000001577_0000001910',
        '2013_05_28_drive_0018_sync/0000000330_0000000543',
        '2013_05_28_drive_0018_sync/0000000002_0000000341',
        '2013_05_28_drive_0018_sync/0000000717_0000000985',
        '2013_05_28_drive_0018_sync/0000000530_0000000727',
        '2013_05_28_drive_0018_sync/0000000975_0000001200',
        '2013_05_28_drive_0018_sync/0000003033_0000003229',
        '2013_05_28_drive_0018_sync/0000003215_0000003513',
        '2013_05_28_drive_0018_sync/0000001878_0000002099',
        '2013_05_28_drive_0018_sync/0000002269_0000002496']}

########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/autonomousvision/kitti360Scripts

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'kittiId',  # An integer ID that is associated with this label for KITTI-360
    # NOT FOR RELEASING

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

# A list of all labels
# NB:
#   Compared to the default KITTI360 implementation, we set all classes to be
#   ignored at train time to IGNORE. Besides, for 3D semantic segmentation, the
#   'sky' class is absent from point labels. So we discard this class from
#   training, which affects all trainIds beyond 10 compared to the default
#   KITTI360 setup.
labels = [
    # name, id, kittiId, trainId, category, catId, hasInstances, ignoreInEval, color
    Label('unlabeled', 0, -1, IGNORE, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, -1, IGNORE, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, -1, IGNORE, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, -1, IGNORE, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, -1, IGNORE, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, -1, IGNORE, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, -1, IGNORE, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 1, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 3, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 2, IGNORE, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 10, IGNORE, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 11, 2, 'construction', 2, True, False, (70, 70, 70)),
    Label('wall', 12, 7, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 8, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 30, IGNORE, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 31, IGNORE, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 32, IGNORE, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 21, 5, 'object', 3, True, False, (153, 153, 153)),
    Label('polegroup', 18, -1, IGNORE, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 23, 6, 'object', 3, True, False, (250, 170, 30)),
    Label('traffic sign', 20, 24, 7, 'object', 3, True, False, (220, 220, 0)),
    Label('vegetation', 21, 5, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 4, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 9, IGNORE, 'sky', 5, False, True, (70, 130, 180)),
    Label('person', 24, 19, 10, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 20, 11, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 12, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 13, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 34, 14, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 16, IGNORE, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 15, IGNORE, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 33, 15, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 16, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 17, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('garage', 34, 12, 2, 'construction', 2, True, False, (64, 128, 128)),
    Label('gate', 35, 6, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('stop', 36, 29, IGNORE, 'construction', 2, True, True, (150, 120, 90)),
    Label('smallpole', 37, 22, 5, 'object', 3, True, False, (153, 153, 153)),
    Label('lamp', 38, 25, IGNORE, 'object', 3, True, False, (0, 64, 64)),
    Label('trash bin', 39, 26, IGNORE, 'object', 3, True, False, (0, 128, 192)),
    Label('vending machine', 40, 27, IGNORE, 'object', 3, True, False, (128, 64, 0)),
    Label('box', 41, 28, IGNORE, 'object', 3, True, False, (64, 64, 128)),
    Label('unknown construction', 42, 35, IGNORE, 'void', 0, False, True, (102, 0, 0)),
    Label('unknown vehicle', 43, 36, IGNORE, 'void', 0, False, True, (51, 0, 51)),
    Label('unknown object', 44, 37, IGNORE, 'void', 0, False, True, (32, 32, 32)),
    Label('license plate', -1, -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# Dictionaries for a fast lookup
NAME2LABEL = {label.name: label for label in labels}
ID2LABEL = {label.id: label for label in labels}
TRAINID2LABEL = {label.trainId: label for label in reversed(labels)}
KITTIID2LABEL = {label.kittiId: label for label in labels}  # KITTI-360 ID to cityscapes ID
CATEGORY2LABELS = {}
for label in labels:
    category = label.category
    if category in CATEGORY2LABELS:
        CATEGORY2LABELS[category].append(label)
    else:
        CATEGORY2LABELS[category] = [label]
KITTI360_NUM_CLASSES = len(TRAINID2LABEL) - 1  # 18
INV_OBJECT_LABEL = {k: TRAINID2LABEL[k].name for k in range(KITTI360_NUM_CLASSES)}
OBJECT_COLOR = np.asarray([TRAINID2LABEL[k].color for k in range(KITTI360_NUM_CLASSES)])
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}
ID2TRAINID = torch.LongTensor([label.trainId for label in labels])


########################################################################
#                                 Utils                                #
########################################################################

def read_kitti360_window(filepath, instance=False, remap=False):
    with open(filepath, "rb") as f:
        window = PlyData.read(f)
        attributes = [p.name for p in window['vertex'].properties]
        pos = torch.stack([torch.FloatTensor(window["vertex"][axis]) for axis in ["x", "y", "z"]], dim=-1)
        rgb = torch.stack([torch.FloatTensor(window["vertex"][axis]) for axis in ["red", "green", "blue"]], dim=-1) / 255
        data = Data(pos=pos, rgb=rgb)
        if 'semantic' in attributes:
            y = torch.LongTensor(window["vertex"]['semantic'])
            data.y = ID2TRAINID[y] if remap else y
        if instance and 'instance' in attributes:
            data.instance = torch.LongTensor(window["vertex"]['instance'])
    return data


def read_variable(fid, name, M, N):
    """ Credit: https://github.com/autonomousvision/kitti360Scripts """
    # rewind
    fid.seek(0, 0)

    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success == 0:
        return None

    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat


def load_intrinsics(intrinsic_file, cam_id=0):
    """ Load KITTI360 perspective camera intrinsics

    Credit: https://github.com/autonomousvision/kitti360Scripts
    """

    intrinsic_loaded = False
    width = -1
    height = -1
    with open(intrinsic_file) as f:
        intrinsics = f.read().splitlines()
    for line in intrinsics:
        line = line.split(' ')
        if line[0] == f'P_rect_0{cam_id}:':
            K = [float(x) for x in line[1:]]
            K = np.reshape(K, [3, 4])
            intrinsic_loaded = True
        elif line[0] == f'R_rect_0{cam_id}:':
            R_rect = np.eye(4)
            R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
        elif line[0] == f"S_rect_0{cam_id}:":
            width = int(float(line[1]))
            height = int(float(line[2]))
    assert (intrinsic_loaded == True)
    assert (width > 0 and height > 0)

    return K, R_rect, width, height


def load_calibration_camera_to_pose(filename):
    """ load KITTI360 camera-to-pose calibration

    Credit: https://github.com/autonomousvision/kitti360Scripts
    """
    Tr = {}
    with open(filename, 'r') as fid:
        cameras = ['image_00', 'image_01', 'image_02', 'image_03']
        lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
        for camera in cameras:
            Tr[camera] = np.concatenate((read_variable(fid, camera, 3, 4), lastrow))
    return Tr


########################################################################
#                                Window                                #
########################################################################

class Window:
    """Small placeholder for point cloud window data."""

    def __init__(self, window_path, sampling_path):
        # Recover useful information from the path
        self.path = window_path
        self.sampling_path = sampling_path
        split, modality, sequence_name, window_name = osp.splitext(window_path)[0].split('/')[-4:]
        self.split = split
        self.modality = modality
        self.sequence = sequence_name
        self.window = window_name

        # Load window data and sampling data
        self._data = torch.load(window_path)
        self._sampling = torch.load(sampling_path)

    @property
    def data(self):
        return self._data

    @property
    def num_points(self):
        return self.data.num_nodes

    @property
    def centers(self):
        return self._sampling['data']

    @property
    def sampling_labels(self):
        return torch.from_numpy(self._sampling['labels'])

    @property
    def sampling_label_counts(self):
        return torch.from_numpy(self._sampling['label_counts'])

    @property
    def sampling_grid_size(self):
        return self._sampling['grid_size']

    @property
    def num_centers(self):
        return self.centers.num_nodes

    def __repr__(self):
        display_attr = ['split', 'sequence', 'window', 'num_points', 'num_centers']
        attr = ', '.join([f'{a}={getattr(self, a)}' for a in display_attr])
        return f'{self.__class__.__name__}({attr})'


########################################################################
#                           KITTI360Cylinder                           #
########################################################################

class KITTI360Cylinder(InMemoryDataset):
    """
    Child class of KITTI360 supporting sampling of 3D cylinders
    within each window.

    When `sample_per_epoch` is specified, indexing the dataset produces
    cylinders randomly picked so as to even-out class distributions.
    When `sample_per_epoch=0`, the cylinders are regularly sampled and
    accessed normally by indexing the dataset.

    http://www.cvlibs.net/datasets/kitti-360/

    Parameters
    ----------
    # TODO: parameters
    """
    num_classes = KITTI360_NUM_CLASSES

    def __init__(
            self, root, split="train", sample_per_epoch=15000, radius=6,
            sample_res=0.3, transform=None, pre_transform=None,
            pre_filter=None, keep_instance=False):

        self._split = split
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._sample_res = sample_res
        self._keep_instance = keep_instance
        self._window = None
        self._window_idx = None

        # Initialization with downloading and all preprocessing
        super(KITTI360Cylinder, self).__init__(
            root, transform, pre_transform, pre_filter)

        # Read all sampling files to prepare for cylindrical sampling.
        # If self.is_random (ie sample_per_eopch > 0), need to recover
        # each window's sampling centers label counts for class-balanced
        # sampling. Otherwise, need to recover the number of cylinders
        # per window for deterministic sampling.
        self._label_counts = torch.zeros(
            len(self.windows), self.num_classes).long()
        self._sampling_sizes = torch.zeros(len(self.windows)).long()
        for i, path in enumerate(self.sampling_paths):

            # Recover the label of cylindrical sampling centers and
            # their count in each window
            sampling = torch.load(path)
            centers = sampling['data']

            # Save the number of sampling cylinders in the window
            self._sampling_sizes[i] = centers.num_nodes

            # If class-balanced sampling is not necessary, skip the rest
            if not self.is_random:
                continue

            # If the data has no labels, class-balanced sampling cannot
            # be performed
            if sampling['labels'] is None:
                raise ValueError(
                    f'Cannot do class-balanced random sampling if data has no '
                    f'labels. Please set sample_per_epoch=0 for test data.')

            # Save the label counts for each window sampling. Cylinders
            # whose center label is IGNORE will not be sampled
            labels = torch.LongTensor(sampling['labels'])
            counts = torch.LongTensor(sampling['label_counts'])
            valid_labels = labels != IGNORE
            labels = labels[valid_labels]
            counts = counts[valid_labels]
            self._label_counts[i, labels] = counts

        if self.is_random:
            assert self._label_counts.sum() > 0, \
                'The dataset has no sampling centers with relevant classes, ' \
                'check that your data has labels, that they follow the ' \
                'nomenclature defined for KITTI360, that your dataset uses ' \
                'enough windows and has reasonable downsampling and cylinder ' \
                'sampling resolutions.'

    @property
    def split(self):
        return self._split

    @property
    def has_labels(self):
        """Self-explanatory attribute needed for BaseDataset."""
        return self.split != 'test'

    @property
    def sample_per_epoch(self):
        """Rules the sampling mechanism for the dataset.

        When `self.sample_per_epoch > 0`, indexing the dataset produces
        random cylindrical sampling, picked so as to even-out the class
        distribution across the dataset.

        When `self.sample_per_epoch <= 0`, indexing the dataset
        addresses cylindrical samples in a deterministic fashion. The
        cylinder indices are ordered with respect to their acquisition
        window and the regular grid sampling of the centers in each
        window.
        """
        return self._sample_per_epoch

    @property
    def is_random(self):
        return self.sample_per_epoch > 0

    @property
    def windows(self):
        """Filenames of the dataset windows."""
        return WINDOWS[self.split]

    @property
    def paths(self):
        """Paths to the dataset windows data."""
        return [osp.join(self.processed_dir, self.split, '3d', f'{p}.pt') for p in self.windows]

    @property
    def sampling_paths(self):
        """Paths to the dataset windows sampling data."""
        return [f'{osp.splitext(p)[0]}_{hash(self._sample_res)}.pt' for p in self.paths]

    @property
    def label_counts(self):
        """Count of cylindrical sampling center of each class, for each
        window. Used for class-balanced random sampling of cylinders in
        the dataset, when self.is_random==True.
        """
        return self._label_counts

    @property
    def sampling_sizes(self):
        """Number of cylindrical sampling, for each window. Used for
        deterministic sampling of cylinders in the dataset, when
        self.is_random==False.
        """
        return self._sampling_sizes

    @property
    def window(self):
        """Currently loaded window."""
        return self._window

    @property
    def window_idx(self):
        """Index of the currently loaded window in self.windows."""
        return self._window_idx

    @property
    def raw_file_names(self):
        """The filepaths to find in order to skip the download."""
        # TODO: add folders for 2D
        # return ['calibration', 'data_2d_raw', 'data_2d_test', 'data_3d_semantics', 'data_3d_semantics_test', 'data_poses']
        return ['data_3d_semantics', 'data_3d_semantics_test']

    @property
    def raw_3d_file_names(self):
        """These are the absolute paths to the raw window files."""
        # The directory where train/test raw scans are
        raw_3d_dir = self.raw_file_names[1] if self.split == 'test' else self.raw_file_names[0]
        return [
            osp.join(self.raw_dir, raw_3d_dir, '/'.split(x)[0], 'static', '/'.split(x)[1] + '.ply')
            for x in self.windows]

    @property
    def processed_3d_file_names(self):
        return [osp.join(split, '3d', f'{p}.pt') for split, w in WINDOWS.items() for p in w]

    @property
    def processed_3d_sampling_file_names(self):
        return [
            osp.join(split, '3d', f'{p}_{hash(self._sample_res)}.pt')
            for split, w in WINDOWS.items() for p in w]

    @property
    def processed_file_names(self):
        """The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing
        """
        # TODO: add 2D files for multimodal
        # [osp.join(split, '2d', p + '.pt') for split, w in WINDOWS.items() for p in w]
        return self.processed_3d_file_names + self.processed_3d_sampling_file_names

    def download(self):
        raise NotImplementedError('KITTI360 automatic download not implemented yet')

    def process(self):
        # TODO: for 2D, can't simply loop over those, need to treat 2D and 3D separately
        for i_window, path in tq(enumerate(self.processed_3d_file_names)):

            # Extract useful information from <path>
            split, modality, sequence_name, window_name = osp.splitext(path)[0].split('/')
            window_path = osp.join(self.processed_dir, path)
            sampling_path = osp.join(
                self.processed_dir, split, modality, sequence_name,
                f'{window_name}_{hash(self._sample_res)}.pt')

            # If required files exist, skip processing
            if osp.exists(window_path) and osp.exists(sampling_path):
                continue

            # Process the window
            if not osp.exists(window_path):

                # If windows sampling data already exists, remove it,
                # because it may be out-of-date
                if osp.exists(sampling_path):
                    os.remove(sampling_path)

                # Create necessary parent folders if need be
                os.makedirs(osp.dirname(window_path), exist_ok=True)

                # Read the raw window data
                data = read_kitti360_window(
                    self.raw_3d_file_names[i_window],
                    instance=self._keep_instance, remap=True)

                # Apply pre_transform
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                # Pre-compute KD-Tree to save time when sampling later
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                data[cT.CylinderSampling.KDTREE_KEY] = tree

                # Save pre_transformed data to the processed dir/<path>
                torch.save(data, window_path)

            else:
                data = torch.load(window_path)

            # Prepare data to build cylinder centers. Only keep 'pos'
            # and 'y' (if any) attributes and drop the z coordinate in
            # 'pos'.
            # NB: we can modify 'data' inplace here to avoid cloning
            for key in data.keys:
                if key not in ['pos', 'y']:
                    delattr(data, key)
            data.pos[:, 2] = 0

            # Compute the sampling of cylinder centers for the window
            sampler = cT.GridSampling3D(size=self._sample_res)
            centers = sampler(data)
            centers.pos = centers.pos[:, :2]
            sampling = {
                'data': centers,
                'labels': None,
                'label_counts': None,
                'grid_size': self._sample_res}

            # If data has labels (ie not test set), save which labels
            # are present in the window and their count. These will be
            # used at sampling time to pick cylinders so as to even-out
            # class distributions
            if hasattr(centers, 'y'):
                unique, counts = np.unique(np.asarray(centers.y), return_counts=True)
                sampling['labels'] = unique
                sampling['label_counts'] = counts

            torch.save(sampling, sampling_path)

    def _load_window(self, idx):
        """Load a window and its sampling data into memory based on its
        index in the self.windows list.
        """
        # Check if the window is not already loaded
        if self.window_idx == idx:
            return

        # Load the window data and associated sampling data
        self._window = Window(self.paths[idx], self.sampling_paths[idx])
        self._window_idx = idx

    def __len__(self):
        return self.sample_per_epoch if self.is_random else self.sampling_sizes.sum()

    def __getitem__(self, idx):
        r"""Gets the cylindrical sample at index `idx` and transforms it
        (in case a `self.transform` is given).

        The expected indexing format depends on `self.is_random`. If
        `self.is_random=True` (ie `self.sample_per_epoch > 0`), then
        `idx` is a tuple carrying `(label, idx_window)` indicating
        which label to pick from which window. If `self.is_random=False`
        then `idx` is an integer in [0, len(self)-1] indicating which
        cylinder to pick among the whole dataset.
        """
        if self.is_random:
            data = self._get_from_label_and_window_idx(*idx)
        else:
            data = self._get_from_global_idx(idx)
        data = data if self.transform is None else self.transform(data)
        return data

    def _get_from_label_and_window_idx(self, label, idx_window):
        """Load a random cylindrical sample of label `ìdx_label` from
        window `idx_window`.
        """
        # Load the associated window
        self._load_window(idx_window)

        # Pick a random center
        valid_centers = torch.where(self.window.centers.y == label)[0]
        idx_center = np.random.choice(valid_centers.numpy())

        # Get the cylindrical sampling
        center = self.window.centers.pos[idx_center]
        sampler = cT.CylinderSampling(self._radius, center, align_origin=False)
        data = sampler(self.window.data)

        # Save the window index and center index in the data. This will
        # be used in the KITTI360Tracker to accumulate per-window votes
        data.idx_window = idx_window
        data.idx_center = idx_center

        return data

    def _get_from_global_idx(self, idx):
        """Load the cylindrical sample of global index `idx`. The global
        indices refer to sampling centers considered in `self.windows`
        order.
        """
        # Split the global idx into idx_window and idx_center
        cum_sizes = self.sampling_sizes.cumsum(0)
        idx_window = torch.bucketize(idx, cum_sizes, right=True)
        offsets = torch.cat((torch.zeros(1), cum_sizes)).long()
        idx_center = idx - offsets[idx_window]

        # Load the associated window
        self._load_window(idx_window)

        # Get the cylindrical sampling
        center = self.window.centers.pos[idx_center]
        sampler = cT.CylinderSampling(self._radius, center, align_origin=False)
        data = sampler(self.window.data)

        # Save the window index and center index in the data. This will
        # be used in the KITTI360Tracker to accumulate per-window votes
        data.idx_window = idx_window
        data.idx_center = idx_center

        return data

    def _pick_random_label_and_window(self):
        """Generates an `(label, idx_window)` tuple as expected by
        `self.__getitem` when `self.is_random=True`.

        This function is typically intended be used by a PyTorch Sampler
        to build a generator to iterate over random samples of the
        dataset while minimizing window loading overheads.
        """
        if not self.is_random:
            raise ValueError('Set sample_per_epoch > 0 for random sampling.')

        # First, pick a class randomly. This guarantees all classes are
        # equally represented. Note that classes are assumed to be all
        # integers in [0, self.num_classes-1] here. Besides, if a class
        # is absent from label_counts (ie no cylinder carries the
        # label), it will not be picked.
        seen_labels = torch.where(self.label_counts.sum(dim=0) > 0)[0]
        label = np.random.choice(seen_labels.numpy())

        # Then, pick a window that has a cylinder with such class, based
        # on class counts.
        counts = self.label_counts[:, label]
        weights = (counts / counts.sum()).numpy()
        idx_window = np.random.choice(range(len(self.windows)), p=weights)

        return label, idx_window


########################################################################
#                              Data splits                             #
########################################################################

class KITTI360Sampler(Sampler):
    """This sampler is responsible for creating KITTICylinder
    `(label, idx_window)` indices for random sampling of cylinders
    across all windows.

    In order to minimize window loading overheads, the KITTI360Sampler
    organizes the samples so that same-window cylinders are queried
    consecutively.
    """

    def __init__(self, dataset):
        # This sampler only makes sense for KITTICylinder datasets
        # implementing random sampling (ie dataset.is_random=True)
        assert dataset.is_random
        self.dataset = dataset

    def __iter__(self):
        # Generate random (label, idx_window) tuple indices
        labels = torch.empty(len(self), dtype=torch.long)
        windows = torch.empty(len(self), dtype=torch.long)
        for i in range(len(self)):
            label, idx_window = self.dataset._pick_random_label_and_window()
            labels[i] = label
            windows[i] = idx_window

        # Shuffle the order in which required windows will be loaded
        unique_windows = windows.unique()
        window_order = unique_windows[torch.randperm(unique_windows.shape[0])]

        # Compute the order in which the cylinders will be loaded
        order = window_order[windows].argsort()

        return iter([(l, w) for l, w in zip(labels[order], windows[order])])

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_samples={len(self)})'


########################################################################
#                            KITTI360Dataset                           #
########################################################################

class KITTI360Dataset(BaseDataset):
    """
    # TODO: comments
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        radius = dataset_opt.get('radius', 6)
        train_sample_res = dataset_opt.get('train_sample_res', 0.3)
        eval_sample_res = dataset_opt.get('eval_sample_res', radius / 2)
        keep_instance = dataset_opt.get('keep_instance', False)
        sample_per_epoch = dataset_opt.get('sample_per_epoch', 15000)
        train_is_trainval = dataset_opt.get('train_is_trainval', False)

        self.train_dataset = KITTI360Cylinder(
            self._data_path,
            radius=radius,
            sample_res=train_sample_res,
            keep_instance=keep_instance,
            sample_per_epoch=sample_per_epoch,
            split='train' if not train_is_trainval else 'trainval',
            pre_transform=self.pre_transform,
            transform=self.train_transform)

        self.val_dataset = KITTI360Cylinder(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
            sample_per_epoch=-1,
            split='val',
            pre_transform=self.pre_transform,
            transform=self.val_transform)

        self.test_dataset = KITTI360Cylinder(
            self._data_path,
            radius=radius,
            sample_res=eval_sample_res,
            keep_instance=keep_instance,
            sample_per_epoch=-1,
            split='test',
            pre_transform=self.pre_transform,
            transform=self.test_transform)

        # A dedicated sampler must be created for the train set. Indeed,
        # self.train_dataset.sample_per_epoch > 0 means cylindrical
        # samples will be picked randomly across all windows. In order
        # to minimize window loading overheads, the train_sampler
        # organizes the epoch batches so that same-window cylinders are
        # queried consecutively.
        self.train_sampler = KITTI360Sampler(self.train_dataset)

        # If a `class_weight_method` is provided in the dataset config,
        # the dataset will have a `weight_classes` to be used when
        # computing the loss
        if dataset_opt.class_weight_method:
            # TODO: find an elegant way of returning class weights for train set
            raise NotImplementedError('KITTI360Dataset does not support class weights yet.')

    @property
    def test_data(self):
        # TODO this needs to change for KITTI360, the raw data will be extracted directly from the files
        return self.test_dataset[0].raw_test_data

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return KITTI360Tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
