import bpy, os, time
sc = bpy.context.scene
W = int(os.environ.get('RW', '1280')); H = int(os.environ.get('RH', '720'))
sc.render.resolution_x = W; sc.render.resolution_y = H; sc.render.resolution_percentage = 100
sc.eevee.taa_render_samples = int(os.environ.get('SAMPLES', '24'))
sc.camera = bpy.data.objects['CAM_projector']
sc.render.image_settings.file_format = 'JPEG'; sc.render.image_settings.quality = 92
sc.render.filepath = os.environ.get('OUT', '/tmp/out/') + 'f_'
sc.frame_start = int(os.environ['FS']); sc.frame_end = int(os.environ['FE'])
t = time.time()
bpy.ops.render.render(animation=True)
print('CHUNK_TIME', time.time() - t)
