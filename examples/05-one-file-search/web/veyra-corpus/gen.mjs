// Deterministic generator for the fictional Station Veyra corpus + question
// set used by the abstention/containment demo (../ablation.mjs,
// public/reproduce-ablation.mjs). Every fact is one markdown file; evidence
// chains are exact fact-id lists. This is the source of truth for
// veyra.pikelet and veyra-ablated.pikelet — the compiled artifacts
// themselves are release assets (see fetch-veyra.mjs), not committed here,
// since a full self-calibrated inline-encoder compile is ~25MB per pack.
// Rebuild: `node gen.mjs && node make-chamber43.mjs`, then compile each
// corpus/ dir with `pikelet compile`.
import fs from 'node:fs';
import path from 'node:path';

const OUT = path.dirname(new URL(import.meta.url).pathname);
const CORPUS = path.join(OUT, 'corpus');
const CORPUS_ABL = path.join(OUT, 'corpus-ablated');

// ---------------- entity tables ----------------
const SUPERVISION = [
  ['velnor', 'Velnor', 'Irena Sol'], ['sarnix', 'Sarnix', 'Tomas Vale'],
  ['okelda', 'Okelda', 'Ludmila Adder'], ['immerra', 'Immerra', 'Irena Sol'],
  ['tovash', 'Tovash', 'Priya Rao'], ['rylet', 'Rylet', 'Halvard Brekke'],
  ['betzel', 'Betzel', 'Irena Sol'], ['corvane', 'Corvane', 'Tomas Vale'],
  ['ushtar', 'Ushtar', 'Nadia Kural'], ['penlow', 'Penlow', 'Wren Ossory'],
];
const HOUSING = [
  ['velnor', 14], ['sarnix', 3], ['okelda', 21], ['immerra', 8], ['tovash', 17],
  ['rylet', 5], ['betzel', 11], ['corvane', 9], ['ushtar', 20], ['penlow', 6],
];
const FLOORS = [[14, 'third'], [3, 'first'], [21, 'third'], [8, 'second'], [17, 'fourth'],
  [5, 'second'], [11, 'first'], [9, 'third'], [20, 'second'], [6, 'fourth'], [2, 'first'], [12, 'third']];
const COOLING = [[14, 'sapphire'], [3, 'glycol'], [21, 'argon'], [8, 'sapphire'], [17, 'ferrofluid'],
  [5, 'glycol'], [11, 'argon'], [9, 'ferrofluid'], [20, 'sapphire'], [6, 'argon'], [2, 'glycol'], [12, 'ferrofluid']];
const CLEARANCE = [['velnor', 'crimson'], ['sarnix', 'cobalt'], ['okelda', 'viridian'], ['immerra', 'crimson'],
  ['tovash', 'ochre'], ['rylet', 'cobalt'], ['betzel', 'viridian'], ['corvane', 'ochre'],
  ['ushtar', 'crimson'], ['penlow', 'cobalt']];
const VAULTS = [['crimson', 'Vault A'], ['cobalt', 'Vault B'], ['viridian', 'Vault C'], ['ochre', 'Vault D']];
const SIBLINGS = [['sib-1', 'Tomas Vale', 'Irena Sol'], ['sib-2', 'Sable Moss', 'Quentin Vesely'],
  ['sib-3', 'Ferro Lindqvist', 'Nadia Kural']];
const BIRTHS = [['irena', 'Irena Sol', 'Trelling Cove'], ['tomas', 'Tomas Vale', 'Ostmark Ridge'],
  ['priya', 'Priya Rao', 'Windmere Flats'], ['halvard', 'Halvard Brekke', 'Corran Deep'],
  ['wren', 'Wren Ossory', 'Salt Hollow'], ['ludmila', 'Ludmila Adder', 'Verak Sound']];
const ROLES = [['sable', 'Sable Moss', 'station quartermaster, responsible for provisioning and stores'],
  ['quentin', 'Quentin Vesely', 'maintainer of the water reclamation loop'],
  ['ferro', 'Ferro Lindqvist', 'calibrator of the seismic monitoring array'],
  ['costin', 'Costin Draney', 'cataloguer of the geological sample archive'],
  ['okonkwo', 'Okonkwo Nkemi', 'coordinator of dock and freight scheduling']];
const HUMIDITY = [[3, 'forty-two'], [8, 'thirty-eight'], [14, 'forty-five'], [17, 'fifty'], [20, 'forty'], [6, 'thirty-six']];
const REVIEWS = [['sarnix', 'Thursday'], ['tovash', 'Monday'], ['rylet', 'Wednesday'], ['corvane', 'Friday'], ['ushtar', 'Tuesday']];

const FILLERS = [
  'The entry was confirmed during the most recent quarterly registry audit and carries no outstanding amendments or pending correction requests.',
  'Station clerks recorded this detail in the operations ledger, and it remains current as of the latest posting cycle on the board.',
  'This record was transcribed from the duty office files and has been countersigned by the registry office without further remark.',
  'The information appears in the standing station documentation, and no revision requests have been filed against it by any department.',
];

const facts = [];
let fi = 0;
function fact(id, title, sentence, paraphrase) {
  const filler = FILLERS[fi++ % FILLERS.length];
  const body = `# ${title}\n\n${sentence} ${paraphrase} ${filler}\n`;
  const words = body.split(/\s+/).filter(Boolean).length;
  if (words < 28) throw new Error(`passage too short (${words}w): ${id}`);
  facts.push({ id, title, body });
}

for (const [k, proj, person] of SUPERVISION)
  fact(`sup-${k}`, `Registry: ${proj} Supervision`,
    `${person} supervises the ${proj} project.`,
    `The ${proj} project operates under the direct supervision of researcher ${person}.`);
for (const [k, ch] of HOUSING) {
  const proj = SUPERVISION.find((s) => s[0] === k)[1];
  fact(`loc-${k}`, `Registry: ${proj} Location`,
    `The ${proj} project is housed in Chamber ${ch}.`,
    `Chamber ${ch} is the assigned working space of the ${proj} project.`);
}
for (const [ch, fl] of FLOORS)
  fact(`floor-ch${ch}`, `Registry: Chamber ${ch} Floor`,
    `Chamber ${ch} is located on the ${fl} floor of the station.`,
    `The ${fl} floor is where Chamber ${ch} sits within the station layout.`);
fact('prop-oxygen', 'Registry: Oxygen Reserves Policy',
  'Only third-floor chambers are fitted with emergency oxygen reserves.',
  'Chambers located on any floor other than the third carry no emergency oxygen reserve fittings at all.');
fact('prop-freight', 'Registry: Freight Access Policy',
  'Only first-floor chambers have direct freight-lift access.',
  'Chambers on the upper floors must route deliveries through the stairwell hand carts instead of the freight lift.');
for (const [ch, cool] of COOLING)
  fact(`cool-ch${ch}`, `Registry: Chamber ${ch} Cooling`,
    `Chamber ${ch} uses ${cool} cooling.`,
    `The cooling system installed in Chamber ${ch} is of the ${cool} type.`);
fact('cert-sapphire', 'Registry: Sapphire Certification',
  'Sapphire-cooled chambers are certified for Class-R experiments.',
  'Certification for Class-R experimental work is granted to chambers that run sapphire cooling systems.');
fact('cert-argon', 'Registry: Argon Certification',
  'Argon-cooled chambers are certified for Class-T experiments.',
  'Certification for Class-T experimental work is granted to chambers that run argon cooling systems.');
for (const [k, clr] of CLEARANCE) {
  const proj = SUPERVISION.find((s) => s[0] === k)[1];
  fact(`clr-${k}`, `Registry: ${proj} Clearance`,
    `The ${proj} project uses ${clr} clearance.`,
    `Members of the ${proj} project carry ${clr} clearance badges while on shift.`);
}
for (const [clr, vault] of VAULTS)
  fact(`vault-${clr}`, `Registry: ${clr[0].toUpperCase() + clr.slice(1)} Vault Access`,
    `${clr[0].toUpperCase() + clr.slice(1)} clearance permits access to ${vault}.`,
    `Holders of ${clr} clearance may enter ${vault} when accompanied by their badge.`);
for (const [id, a, b] of SIBLINGS)
  fact(id, `Registry: Family Note ${id.toUpperCase()}`,
    `${a} and ${b} are siblings.`,
    `The station family register lists ${a} as the sibling of ${b}.`);
for (const [k, person, place] of BIRTHS)
  fact(`born-${k}`, `Registry: ${person} Origin`,
    `${person} was born in ${place}.`,
    `The personnel file gives ${place} as the recorded birthplace of ${person}.`);
for (const [k, person, role] of ROLES)
  fact(`role-${k}`, `Registry: ${person} Duties`,
    `${person} serves as the ${role}.`,
    `Day-to-day duty rosters list ${person} against that responsibility.`);
for (const [ch, pct] of HUMIDITY)
  fact(`hum-ch${ch}`, `Registry: Chamber ${ch} Humidity`,
    `Chamber ${ch} is held at ${pct} percent relative humidity.`,
    `Environmental control keeps Chamber ${ch} steady at the ${pct} percent setting.`);
for (const [k, day] of REVIEWS) {
  const proj = SUPERVISION.find((s) => s[0] === k)[1];
  fact(`rev-${k}`, `Registry: ${proj} Review Day`,
    `The ${proj} team holds its weekly review on ${day}s.`,
    `${day} is the fixed weekly review slot for the ${proj} project.`);
}
fact('store-ch2', 'Registry: Chamber 2 Use',
  'Chamber 2 serves as dry storage for filtration spares.',
  'Racking in Chamber 2 holds the spare filtration cartridges and gasket stock.');
fact('store-ch12', 'Registry: Chamber 12 Use',
  'Chamber 12 houses the backup centrifuge assembly.',
  'The disassembled backup centrifuge is kept under covers in Chamber 12.');
fact('mess-1', 'Registry: Mess Rotation One',
  'The mess hall rotates its soup selection every third day.',
  'Kitchen staff post the upcoming soup rotation on the corridor board each cycle.');
fact('mess-2', 'Registry: Mess Rotation Two',
  'Bread service in the mess hall alternates between rye and barley loaves.',
  'The bakery alternates its two loaf styles across successive service days.');
fact('maint-1', 'Registry: Corridor Lighting',
  'Corridor lighting panels are inspected on a rolling twelve-day cycle.',
  'Maintenance walks the corridor lighting circuit across a rolling twelve-day inspection loop.');
fact('maint-2', 'Registry: Air Filter Rotation',
  'Air handling filters are swapped by the maintenance crew on alternating shifts.',
  'Filter swaps for the air handlers alternate between the two maintenance shifts.');
fact('maint-3', 'Registry: Dock Seal Checks',
  'Dock seal integrity checks are logged after every freight arrival.',
  'Each arriving freight run triggers a fresh seal integrity entry in the dock log.');

// ---------------- questions ----------------
const Q = [];
const q = (id, cls, text, answer, type, evidence, extra = {}) =>
  Q.push({ id, cls, q: text, answer, type, evidence, ...extra });

// direct
q('d01', 'direct', 'Who supervises the Velnor project?', 'Irena Sol', 'name', ['sup-velnor']);
q('d02', 'direct', 'Who supervises the Okelda project?', 'Ludmila Adder', 'name', ['sup-okelda']);
q('d03', 'direct', 'Who supervises the Penlow project?', 'Wren Ossory', 'name', ['sup-penlow']);
q('d04', 'direct', 'Who supervises the Corvane project?', 'Tomas Vale', 'name', ['sup-corvane']);
q('d05', 'direct', 'Which chamber houses the Immerra project?', 'Chamber 8', 'exact', ['loc-immerra']);
q('d06', 'direct', 'Which chamber houses the Rylet project?', 'Chamber 5', 'exact', ['loc-rylet']);
q('d07', 'direct', 'Which chamber houses the Ushtar project?', 'Chamber 20', 'exact', ['loc-ushtar']);
q('d08', 'direct', 'What cooling does Chamber 14 use?', 'sapphire', 'exact', ['cool-ch14']);
q('d09', 'direct', 'What cooling does Chamber 6 use?', 'argon', 'exact', ['cool-ch6']);
q('d10', 'direct', 'On which floor is Chamber 17?', 'fourth', 'exact', ['floor-ch17']);
q('d11', 'direct', 'On which floor is Chamber 11?', 'first', 'exact', ['floor-ch11']);
q('d12', 'direct', 'Where was Priya Rao born?', 'Windmere Flats', 'exact', ['born-priya']);
q('d13', 'direct', 'Where was Halvard Brekke born?', 'Corran Deep', 'exact', ['born-halvard']);
q('d14', 'direct', 'What clearance does the Sarnix project use?', 'cobalt', 'exact', ['clr-sarnix']);
q('d15', 'direct', 'What clearance does the Betzel project use?', 'viridian', 'exact', ['clr-betzel']);
q('d16', 'direct', 'Which vault does crimson clearance permit access to?', 'Vault A', 'exact', ['vault-crimson']);
q('d17', 'direct', "Who is Nadia Kural's sibling at the station?", 'Ferro Lindqvist', 'name', ['sib-3']);
q('d18', 'direct', "What is Sable Moss's role at the station?", 'quartermaster', 'exact', ['role-sable']);
q('d19', 'direct', 'On what day does the Sarnix team hold its weekly review?', 'Thursday', 'exact', ['rev-sarnix']);
q('d20', 'direct', 'What is Chamber 2 used for?', 'dry storage', 'exact', ['store-ch2']);

// multihop
q('m01', 'multihop', 'On which floor is the project supervised by Priya Rao housed?', 'fourth', 'exact',
  ['sup-tovash', 'loc-tovash', 'floor-ch17'], { hops: 3 });
q('m02', 'multihop', "On which floor is Ludmila Adder's project housed?", 'third', 'exact',
  ['sup-okelda', 'loc-okelda', 'floor-ch21'], { hops: 3 });
q('m03', 'multihop', 'On which floor is the Sarnix project housed?', 'first', 'exact',
  ['loc-sarnix', 'floor-ch3'], { hops: 2 });
q('m04', 'multihop', 'Does the chamber housing the Corvane project have emergency oxygen reserves?', 'yes', 'yesno',
  ['loc-corvane', 'floor-ch9', 'prop-oxygen'], { hops: 3 });
q('m05', 'multihop', "Does the project supervised by Wren Ossory have emergency oxygen reserves in its chamber?", 'no', 'yesno',
  ['sup-penlow', 'loc-penlow', 'floor-ch6', 'prop-oxygen'], { hops: 4 });
q('m06', 'multihop', "Does Halvard Brekke's project have direct freight-lift access?", 'no', 'yesno',
  ['sup-rylet', 'loc-rylet', 'floor-ch5', 'prop-freight'], { hops: 4 });
q('m07', 'multihop', 'Does the Sarnix project have direct freight-lift access?', 'yes', 'yesno',
  ['loc-sarnix', 'floor-ch3', 'prop-freight'], { hops: 3 });
q('m08', 'multihop', 'Is Irena Sol supervising any project certified for Class-R experiments?', 'yes', 'yesno',
  ['sup-velnor', 'loc-velnor', 'cool-ch14', 'cert-sapphire'],
  { hops: 4, evidenceAlt: [['sup-immerra', 'loc-immerra', 'cool-ch8', 'cert-sapphire']] });
q('m09', 'multihop', "Is Nadia Kural's project certified for Class-R experiments?", 'yes', 'yesno',
  ['sup-ushtar', 'loc-ushtar', 'cool-ch20', 'cert-sapphire'], { hops: 4 });
q('m10', 'multihop', "What experiment class is the Okelda project's chamber certified for?", 'Class-T', 'exact',
  ['loc-okelda', 'cool-ch21', 'cert-argon'], { hops: 3 });
q('m11', 'multihop', 'Which vault can the Tovash project access?', 'Vault D', 'exact',
  ['clr-tovash', 'vault-ochre'], { hops: 2 });
q('m12', 'multihop', "Which vault can Priya Rao's project access?", 'Vault D', 'exact',
  ['sup-tovash', 'clr-tovash', 'vault-ochre'], { hops: 3 });
q('m13', 'multihop', 'Which vault can the Betzel project access?', 'Vault C', 'exact',
  ['clr-betzel', 'vault-viridian'], { hops: 2 });
q('m14', 'multihop', "Do any of Tomas Vale's projects use ochre clearance?", 'yes', 'yesno',
  ['sup-corvane', 'clr-corvane'], { hops: 2 });
q('m15', 'multihop', 'Which project supervisors have a sibling working at the station?', 'Irena Sol, Tomas Vale, Nadia Kural', 'nameset',
  ['sib-1', 'sib-3'], { hops: 4, aggregate: true });
q('m16', 'multihop', 'Does the chamber housing the Immerra project have emergency oxygen reserves?', 'no', 'yesno',
  ['loc-immerra', 'floor-ch8', 'prop-oxygen'], { hops: 3 });
q('m17', 'multihop', "On which floor is Wren Ossory's project housed?", 'fourth', 'exact',
  ['sup-penlow', 'loc-penlow', 'floor-ch6'], { hops: 3 });
q('m18', 'multihop', "What cooling system does the Corvane project's chamber use?", 'ferrofluid', 'exact',
  ['loc-corvane', 'cool-ch9'], { hops: 2 });
q('m19', 'multihop', 'Is the Penlow project certified for Class-T experiments?', 'yes', 'yesno',
  ['loc-penlow', 'cool-ch6', 'cert-argon'], { hops: 3 });
q('m20', 'multihop', 'Which vault can the Velnor project access?', 'Vault A', 'exact',
  ['clr-velnor', 'vault-crimson'], { hops: 2 });

// counting
q('c01', 'counting', 'How many projects does Irena Sol supervise?', '3', 'number',
  ['sup-velnor', 'sup-immerra', 'sup-betzel']);
q('c02', 'counting', 'How many projects does Tomas Vale supervise?', '2', 'number',
  ['sup-sarnix', 'sup-corvane']);
q('c03', 'counting', 'How many projects use cobalt clearance?', '3', 'number',
  ['clr-sarnix', 'clr-rylet', 'clr-penlow']);
q('c04', 'counting', 'How many projects use crimson clearance?', '3', 'number',
  ['clr-velnor', 'clr-immerra', 'clr-ushtar']);
q('c05', 'counting', 'How many chambers are on the third floor?', '4', 'number',
  ['floor-ch14', 'floor-ch21', 'floor-ch9', 'floor-ch12']);
q('c06', 'counting', 'How many chambers use sapphire cooling?', '3', 'number',
  ['cool-ch14', 'cool-ch8', 'cool-ch20']);
q('c07', 'counting', 'How many sibling pairs work at the station?', '3', 'number',
  ['sib-1', 'sib-2', 'sib-3']);
q('c08', 'counting', 'How many projects does Wren Ossory supervise?', '1', 'number', ['sup-penlow']);
q('c09', 'counting', 'How many chambers use glycol cooling?', '3', 'number',
  ['cool-ch3', 'cool-ch5', 'cool-ch2']);
q('c10', 'counting', 'How many projects are housed on the fourth floor?', '2', 'number',
  ['loc-tovash', 'floor-ch17', 'loc-penlow', 'floor-ch6']);

// unsupported (plausible, in-domain, absent)
const U = [
  ['u01', 'Where did Irena Sol attend university?'],
  ['u02', 'Who is Tomas Vale married to?'],
  ['u03', "What is the Velnor project's annual budget?"],
  ['u04', 'In what year was the Okelda project founded?'],
  ['u05', 'What is the ceiling height of Chamber 14?'],
  ['u06', 'How old is Priya Rao?'],
  ['u07', 'What languages does Wren Ossory speak?'],
  ['u08', "What is Halvard Brekke's salary?"],
  ['u09', 'How many staff members work on the Immerra project?'],
  ['u10', 'What color are the walls of Chamber 5?'],
  ['u11', 'Where was Nadia Kural born?'],
  ['u12', 'Where was Sable Moss born?'],
  ['u13', "What clearance does Ludmila Adder personally hold?"],
  ['u14', 'What is the emergency oxygen reserve capacity of Chamber 9?'],
  ['u15', 'On what day does the Velnor team hold its weekly review?'],
  ['u16', 'What cooling does Chamber 4 use?'],
  ['u17', 'Who supervises the Marlow project?'],
  ['u18', 'What vault does amber clearance permit access to?'],
  ['u19', "What is Costin Draney's sibling's name?"],
  ['u20', 'On which floor is Chamber 7?'],
];
for (const [id, text] of U) q(id, 'unsupported', text, null, 'abstain', []);

// off-domain
const O = [
  ['o01', "What is the station's opinion of purple elephants?"],
  ['o02', 'Calculate the orbital velocity of toaster number seven.'],
  ['o03', 'How do I configure a Kubernetes ingress controller?'],
  ['o04', 'What is the capital of France?'],
  ['o05', 'Who won the 1998 football World Cup?'],
  ['o06', 'What is the boiling point of water at sea level?'],
  ['o07', 'Write a haiku about autumn leaves.'],
  ['o08', 'What is the square root of 214?'],
  ['o09', 'What are the side effects of ibuprofen?'],
  ['o10', 'Summarize the plot of Hamlet.'],
];
for (const [id, text] of O) q(id, 'offdomain', text, null, 'abstain', []);

// near-miss counterfactuals
q('n01', 'nearmiss', 'Can the Sarnix project access Vault D?', 'no', 'noevidence', ['clr-sarnix', 'vault-ochre']);
q('n02', 'nearmiss', 'Can the Velnor project access Vault B?', 'no', 'noevidence', ['clr-velnor', 'vault-cobalt']);
q('n03', 'nearmiss', 'Can the Betzel project access Vault A?', 'no', 'noevidence', ['clr-betzel', 'vault-crimson']);
q('n04', 'nearmiss', 'Is Chamber 14 certified for Class-T experiments?', 'no', 'noevidence', ['cool-ch14', 'cert-argon']);
q('n05', 'nearmiss', 'Does viridian clearance permit access to Vault A?', 'no', 'noevidence', ['vault-viridian']);
q('n06', 'nearmiss', "Is Tomas Vale Priya Rao's brother?", 'no', 'noevidence', ['sib-1']);
q('n07', 'nearmiss', 'Does the Ushtar project use ochre clearance?', 'no', 'noevidence', ['clr-ushtar']);
q('n08', 'nearmiss', 'Is the Rylet project housed in Chamber 9?', 'no', 'noevidence', ['loc-rylet']);
q('n09', 'nearmiss', 'Was Wren Ossory born in Trelling Cove?', 'no', 'noevidence', ['born-wren']);
q('n10', 'nearmiss', 'Does Chamber 21 have direct freight-lift access?', 'no', 'noevidence', ['floor-ch21', 'prop-freight']);

// ablation plan: facts removed from the ablated pack, and the questions re-run on it
const ABLATION = {
  removed: ['loc-tovash', 'floor-ch9', 'cert-sapphire', 'vault-ochre', 'sup-okelda', 'sup-betzel'],
  reruns: [
    { qid: 'm01', removedFact: 'loc-tovash', expect: 'abstain' },
    { qid: 'm04', removedFact: 'floor-ch9', expect: 'abstain' },
    { qid: 'm09', removedFact: 'cert-sapphire', expect: 'abstain' },
    { qid: 'm11', removedFact: 'vault-ochre', expect: 'abstain' },
    { qid: 'd02', removedFact: 'sup-okelda', expect: 'abstain' },
    { qid: 'c01', removedFact: 'sup-betzel', expect: '2' },
  ],
};

// ---------------- emit ----------------
for (const dir of [CORPUS, CORPUS_ABL]) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
}
const removedSet = new Set(ABLATION.removed);
for (const f of facts) {
  fs.writeFileSync(path.join(CORPUS, `${f.id}.md`), f.body);
  if (!removedSet.has(f.id)) fs.writeFileSync(path.join(CORPUS_ABL, `${f.id}.md`), f.body);
}
fs.writeFileSync(path.join(OUT, 'questions.json'), JSON.stringify({ questions: Q, ablation: ABLATION }, null, 1));
console.log(`facts: ${facts.length} (ablated corpus: ${facts.length - removedSet.size})`);
console.log(`questions: ${Q.length} — ` + ['direct', 'multihop', 'counting', 'unsupported', 'offdomain', 'nearmiss']
  .map((c) => `${c}:${Q.filter((x) => x.cls === c).length}`).join(' '));
