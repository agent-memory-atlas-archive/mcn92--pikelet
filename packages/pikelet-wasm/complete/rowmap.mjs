// The row map (segment kind 8, spec/COMPLETE_PROFILE.md 3.9): when the index
// holds several vectors per corpus record — the record's own vector plus its
// sentence windows — the sketch's rows are no longer the corpus's records.
// Each record owns a contiguous run of rows, in record order:
//
//   [0,4)   u32 version = 1
//   [4,8)   u32 records         MUST equal manifest corpus.records
//   [8,12)  u32 rows            MUST equal the index segment's row count
//   [12,16) reserved, zero
//   offsets u32[records + 1]    record i owns rows [offsets[i], offsets[i+1])
//
// offsets[0] is 0, offsets[records] is rows, and every record owns at least
// one row (offsets strictly increase), so every row maps to exactly one
// record and every record is reachable from the index. Shared by the builder
// and the reader so the layout has one definition. Environment-neutral.

const HEADER = 16;

export function buildRowMapSegment(offsets) {
  if (!offsets || typeof offsets.length !== 'number' || offsets.length < 2) {
    throw new Error('row map needs offsets for at least one record');
  }
  const records = offsets.length - 1;
  if (offsets[0] !== 0) throw new Error('row map offsets must start at 0');
  for (let i = 0; i < records; i++) {
    if (!Number.isInteger(offsets[i + 1]) || offsets[i + 1] <= offsets[i] || offsets[i + 1] > 0xffffffff) {
      throw new Error(`row map record ${i} must own at least one row`);
    }
  }
  const bytes = new Uint8Array(HEADER + 4 * (records + 1));
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 1, true);
  view.setUint32(4, records, true);
  view.setUint32(8, offsets[records], true);
  for (let i = 0; i <= records; i++) view.setUint32(HEADER + 4 * i, offsets[i], true);
  return bytes;
}

// Validates a row map against the corpus record count and the index's row
// count, returning the offsets and a row -> record table.
export function parseRowMapSegment(bytes, records, rows) {
  if (!(bytes instanceof Uint8Array) || bytes.length < HEADER) throw new Error('.pikelet row map is too short');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 1) throw new Error(`unsupported row map version ${view.getUint32(0, true)}`);
  if (view.getUint32(4, true) !== records) {
    throw new Error(`.pikelet row map covers ${view.getUint32(4, true)} records, corpus declares ${records}`);
  }
  if (view.getUint32(8, true) !== rows) {
    throw new Error(`.pikelet row map covers ${view.getUint32(8, true)} rows, index holds ${rows}`);
  }
  if (view.getUint32(12, true) !== 0) throw new Error('.pikelet row map reserved word is not zero');
  if (bytes.length !== HEADER + 4 * (records + 1)) throw new Error('.pikelet row map length disagrees with its record count');
  const offsets = new Uint32Array(records + 1);
  for (let i = 0; i <= records; i++) offsets[i] = view.getUint32(HEADER + 4 * i, true);
  if (offsets[0] !== 0 || offsets[records] !== rows) throw new Error('.pikelet row map offsets do not span the index');
  const owner = new Uint32Array(rows);
  for (let r = 0; r < records; r++) {
    if (offsets[r + 1] <= offsets[r]) throw new Error(`.pikelet row map gives record ${r} no rows`);
    owner.fill(r, offsets[r], offsets[r + 1]);
  }
  return { offsets, owner, rows };
}
