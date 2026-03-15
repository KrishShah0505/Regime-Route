from fastapi import APIRouter
from strategies.router import RegimeRouter, STRATEGY_REGISTRY, DEFAULT_REGIME_MAP

router = APIRouter()

@router.get("")
async def list_strategies():
    r = RegimeRouter()
    return r.list_strategies()

@router.get("/regime-map")
async def get_regime_map():
    return DEFAULT_REGIME_MAP

@router.get("/{name}")
async def get_strategy(name: str):
    if name not in STRATEGY_REGISTRY:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    cls = STRATEGY_REGISTRY[name]
    return {
        "name": name,
        "description": cls.description,
        "preferred_regimes": cls.preferred_regimes,
        "default_params": cls.default_params,
    }
