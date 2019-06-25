# run once for fitting the DEM model
pcaPaperRunAll:
	python3 smallMri.py 1 1 1

# load previously computed results and generate figures 
pcaPaperLoadAllGenFigs:
	python3 smallMri.py 0 1 1

pcaPaperBmConf:
	python3 smallMriBMConf.py 1 1 1

cogDEMRunAll:
	python3 cognitive_dem.py 1 1 1

cogDEMLoadAllGenFigs:
	python3 cognitive_dem.py 0 1 1

cogDEM_BmConf:
	python3 cognitive_dem_bmConf.py 1 1 1

cogDEM_BmConf:
	python3 cognitive_dem_bmConf.py 1 1 1

