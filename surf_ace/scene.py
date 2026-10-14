"""Document model: objects, selection, undo/redo and change notification."""

from __future__ import annotations

from typing import Callable

OBJECTS = "objects"
GEOMETRY = "geometry"
SELECTION = "selection"


class Scene:
    def __init__(self, max_undo: int = 100):
        self.objects: dict[int, object] = {}
        self.selected: int | None = None
        self.selected_cp: tuple | None = None
        self._next_id = 1
        self._undo: list = []
        self._redo: list = []
        self._max_undo = max_undo
        self._listeners: list[Callable[[str, int | None], None]] = []
        self.modified = False

    # ------------------------------------------------------------------ notification
    def subscribe(self, fn: Callable[[str, int | None], None]) -> None:
        self._listeners.append(fn)

    def notify(self, kind: str, oid: int | None = None) -> None:
        if kind != SELECTION:
            self.modified = True
        for fn in list(self._listeners):
            fn(kind, oid)

    # ------------------------------------------------------------------ undo
    def _snapshot(self):
        return ({k: v.copy() for k, v in self.objects.items()}, self.selected, self._next_id)

    def checkpoint(self) -> None:
        """Call before any modification."""
        self._undo.append(self._snapshot())
        if len(self._undo) > self._max_undo:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, snap) -> None:
        objs, sel, nid = snap
        self.objects = {k: v.copy() for k, v in objs.items()}
        self.selected = sel if sel in self.objects else None
        self.selected_cp = None
        self._next_id = nid
        self.notify(OBJECTS)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    # ------------------------------------------------------------------ editing
    def add(self, obj, select: bool = True, checkpoint: bool = True) -> int:
        if checkpoint:
            self.checkpoint()
        oid = self._next_id
        self._next_id += 1
        self.objects[oid] = obj
        if select:
            self.selected, self.selected_cp = oid, None
        self.notify(OBJECTS)
        return oid

    def replace(self, oid: int, obj, checkpoint: bool = True) -> None:
        if checkpoint:
            self.checkpoint()
        self.objects[oid] = obj
        if self.selected == oid:
            self.selected_cp = None
        self.notify(GEOMETRY, oid)

    def remove(self, oid: int) -> None:
        self.checkpoint()
        self.objects.pop(oid, None)
        if self.selected == oid:
            self.selected, self.selected_cp = None, None
        self.notify(OBJECTS)

    def clear(self) -> None:
        self.objects.clear()
        self.selected = self.selected_cp = None
        self._undo.clear()
        self._redo.clear()
        self.notify(OBJECTS)
        self.modified = False

    def select(self, oid: int | None, cp: tuple | None = None) -> None:
        if oid == self.selected and cp == self.selected_cp:
            return
        self.selected, self.selected_cp = oid, cp
        self.notify(SELECTION, oid)

    def get(self, oid):
        return self.objects.get(oid)

    def surfaces(self) -> dict[int, object]:
        return {k: v for k, v in self.objects.items() if v.kind == "surface"}

    def unique_name(self, base: str) -> str:
        names = {o.name for o in self.objects.values()}
        i = 1
        while f"{base} {i}" in names:
            i += 1
        return f"{base} {i}"
