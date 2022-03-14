from typing import Any, Dict
from models.network_sr import UDKE

def init_net(opt: Dict[str, Any]) -> UDKE:
    opt_net = opt['netG']

    netG = UDKE(n_iter=opt_net['n_iter'],
               in_nc=opt_net['in_nc'],
               nc_x=opt_net['nc_x'],
               out_nc=opt_net['out_nc'],
               nb=opt_net['nb'],
               k_size=opt['data']['k_size'])

    return netG
