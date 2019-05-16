# -*- mode: python -*-

block_cipher = None


a = Analysis(['gui.py'],
             pathex=['C:\\Users\\Philip\\Documents\\GitHub\\LCI-Clustering'],
             binaries=[],
             datas=[ ('UI', 'UI'), ('ttkthemes', 'ttkthemes'),
              ('clustering.py', '.') ],
             hiddenimports=['ttkthemes', 'UI', 'sklearn.utils._cython_blas',
             'sklearn.neighbors.typedefs', 'sklearn.externals.joblib',
             'sklearn.neighbors.quad_tree', 'sklearn.tree',
             'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt5', 'PyQt4'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='HDBSCAN_Clustering_v0.4.0',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='HDBSCAN_Clustering')
