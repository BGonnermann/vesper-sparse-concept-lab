"""Campaign budget and result selection contracts."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class CampaignTests(unittest.TestCase):
    def test_deadline_reserve_rejects_late_launch(self):
        self.assertTrue((ROOT/'scripts/ncp_campaign.py').exists(),'Missing campaign controller')
        import ncp_campaign as campaign
        self.assertFalse(campaign.can_launch(2099))
        self.assertTrue(campaign.can_launch(2101))

    def test_collapse_is_not_promotable(self):
        self.assertTrue((ROOT/'scripts/ncp_campaign.py').exists(),'Missing campaign controller')
        import ncp_campaign as campaign
        good={'status':'completed','metrics':{'val_bpb':.62},'ncp_health':{'collapsed':False}}
        self.assertTrue(campaign.promising(good,.635))
        self.assertFalse(campaign.promising(dict(good,ncp_health={'collapsed':True}),.635))
        self.assertFalse(campaign.promising(dict(good,status='failed'),.635))


if __name__=='__main__': unittest.main()
