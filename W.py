from omero.gateway import *
from omero.model import *
from omero.rtypes import *
from omero.util.tiles import *
from numpy import fromfunction

user = 'facility_staff_1'
pw = 'facility_staff_1_pw'
host = '192.168.56.101'

conn = BlitzGateway(username=user,
                    passwd=pw,
                    host=host,
                    port=4064)
conn.connect()

sizeX = 4096
sizeY = 4096
sizeZ = 1
sizeC = 1
sizeT = 1
tileWidth = 1024
tileHeight = 1024
imageName = "testStitchBig4K-1Ktiles"
description = None
tile_max = 255

pixelsService = conn.getPixelsService()
queryService = conn.getQueryService()

# query = "from PixelsType as p where p.value='int8'"
query = "from PixelsType as p where p.value='float'"
pixelsType = queryService.findByQuery(query, None)
channelList = range(sizeC)
bytesPerPixel = pixelsType.bitSize.val / 8
iId = pixelsService.createImage(
   sizeX,
   sizeY,
   sizeZ,
   sizeT,
   channelList,
   pixelsType,
   imageName,
   description,
   conn.SERVICE_OPTS)

image = conn.getObject("Image", iId)
pid = image.getPixelsId()

def f(x, y):
    """
    create some fake pixel data tile (2D numpy array)
    """
    return (x * y)/(1 + x + y)

def mktile(w, h):
    tile = fromfunction(f, (w, h))
    # tile = tile.astype(int)
    tile = tile.astype(float)
    tile[tile > tile_max] = tile_max
    return list(tile.flatten())

# tile = fromfunction(f, (tileWidth, tileHeight)).astype(int)
tile = fromfunction(f, (tileWidth, tileHeight)).astype(float)
tile_min = float(tile.min())
tile_max = min(tile_max, float(tile.max()))


class Iteration(TileLoopIteration):

    def run(self, data, z, c, t, x, y, tileWidth, tileHeight, tileCount):
        tile2d = mktile(tileWidth, tileHeight)
        data.setTile(tile2d, z, c, t, x, y, tileWidth, tileHeight)

loop = RPSTileLoop(conn.c.sf, PixelsI(pid, False))
loop.forEachTile(256, 256, Iteration())

c = 0
pixelsService.setChannelGlobalMinMax(
   pid, c, tile_min, tile_max, conn.SERVICE_OPTS)

conn._closeSession()

print("Image", iId.val)
