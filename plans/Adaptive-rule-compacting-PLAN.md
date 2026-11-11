# Adaptacyjne kompaktowanie reguł w ACS2VCP — plan dojścia do metody, która realnie pomaga

## Kontekst

**Problem.** ACS2 z HER i VCP produkuje ogromne populacje reguł i przez to liczy się długo.

- Zemło (2025): populacje ACS2HER/ACS2VCP są 2,5–24 razy większe niż ACS2/ACS2ER (Maze6: ok. 4200 wobec 420; MazeB: ok. 7500, maksimum ok. 10 000 i wciąż rośnie w ostatniej próbie). Wiarygodnych jest tylko 15–35% reguł w Maze6 i MazeB. Czas rośnie kilkukrotnie i wraz z liczbą głowic. Wniosek końcowy pracy: zrównoleglić głowice i zmniejszyć populację „np. CRACS”.
- W tym repo dopasowanie reguł to skan liniowy (`Population::form_match_set`), więc koszt jest proporcjonalny do rozmiaru populacji: 10 formowań zbioru na interakcję plus K formowań na każdą propozycję priorytetu. Udokumentowane przebiegi: 950–1700 reguł na głowicę po 2 mln interakcji, wariant z 4 głowicami ok. 2,5 raza droższy od kontroli bez priorytetów (Maze7: 1014 s wobec 402 s).

**Co istnieje i czego brakuje.**

- CRACS (Orhand 2023) działa raz, po uczeniu, i nie patrzy na predykcję nagrody. Autorzy sami wskazują to jako brak. Był sprawdzany na BACS, PEPACS i BEACS, nigdy na zwykłym ACS2, dla którego zostaje z niego sama subsumpcja.
- Choczaj (2024) dodał wariant „dynamiczny”: w trakcie uczenia usuwa reguły o doświadczeniu i jakości poniżej stałych progów, gdy populacja przekracza trzeci próg. Progi są strojone ręcznie dla każdego środowiska. Metoda redukuje też reguły wiarygodne (Maze5: 239 → 157), nie mierzy utraty wiedzy i w części środowisk nie działa. Kod odbiega od artykułu: test subsumpcji jest praktycznie martwy, a filtr doświadczenia obejmuje wszystkie reguły, więc usuwa właśnie te, które dopiero się uczą (`docs/ARCHITECTURE.md`: „Do not port from it”).
- W repo CRACS jest czystą operacją (`acs2-core/src/compaction/`), której nic nie woła.
- Nikt nie rozstrzyga, **kiedy i jak mocno** kompaktować w trakcie uczenia, ani nie chroni tego, od czego zależy VCP: wartości (a przez nie priorytetów) i różnorodności głowic.

**Cel planu.** Nie zakładać z góry jednej metody, tylko przejść lejek: najpierw zmierzyć, co w populacji da się usunąć, potem sprawdzać kandydatów od najtańszego testu do najdroższego, z bramką po każdym etapie. Etapy rozwijają kroki 1–7 z `docs/RULE_COMPACTION.md`.

**Możliwy wkład własny względem CRACS i Choczaja:** redukcja dowodliwie neutralna dla wartości i priorytetów; przycinanie reguł bezczynnych na podstawie pamięci doświadczeń; sterowanie siłą redukcji sygnałem odtwarzania usuniętych reguł; analiza tego, jak kompaktowanie oddziałuje z priorytetyzacją zgodności.

## Zasady, na których opieram rekomendację

1. **Bezpieczeństwo w operatorze, ekonomia w wyzwalaczu.** Sygnał „uczenie się ustabilizowało” zawodzi (znany błąd tego projektu), więc redukcja musi być bezpieczna także wtedy, gdy wyzwalacz odpali w złym momencie.
2. **Reguła pełni trzy role:** model (antycypacja), wartość (niesie maksimum `q·r`), odkrywanie (znaczniki i rodzic przyszłych specjalizacji). Usunięcie jest bezpieczne, gdy każdą z ról przejmuje reguła, która zostaje.
3. **Wartość głowicy to maksimum.** `Q_i(s,a) = max q·r` po zbiorze akcji (`acs2vcp/priority.rs::action_value`). Jeśli ocalała reguła pokrywa te same stany i ma fitness nie mniejszy, to `Q_i` nie zmienia się w żadnym stanie, a więc wariancja między głowicami i priorytety VCP zostają identyczne. To daje dowodliwą gwarancję zamiast pomiaru po fakcie.
4. **W zadaniach z celem jedna populacja niesie model (niezależny od celu) i wartości (zależne od celu).** Reguła zbędna dla modelu nie musi być zbędna dla wartości, dlatego CRACS ślepy na nagrodę jest tu ryzykowny.
5. **Pamięć doświadczeń to dowód.** ER/HER/VCP mają zapis prawdziwych przejść: można na nim sprawdzać skutki redukcji i wskazywać reguły bezczynne, bez dostępu do środowiska.
6. **Liczy się trwałość.** Redukcja pomaga tylko wtedy, gdy usunięte reguły nie wracają. Odsetek odtworzonych reguł to najtańszy wewnętrzny sygnał szkody.

## Możliwości

### A. Co usuwać (operator)

| | Operator | Co usuwa | Gwarancja | Uwagi |
|---|---|---|---|---|
| A1 | CRACS z artykułu (jest w repo) | regułę zawartą w ogólniejszej o tej samej akcji i efekcie (`q₂ > θ_r` lub `q₂ ≥ q₁`, zgodne znaczniki) | wiedza zachowana dokładnie (testy w repo) | ślepy na nagrodę; w trakcie uczenia subsumer może być niedojrzały |
| A2 | Subsumpcja wsteczna „dojrzała” | to samo, ale subsumer musi spełniać kryterium samego ACS2: wiarygodny, `exp > θ_exp`, nieoznaczony | usunięta reguła i tak byłaby dziś odrzucona przy wstawianiu, więc nie wraca, dopóki subsumer pozostaje dojrzały | mniejsza kompresja |
| A3 | Strażnik wartości `δ` (nakładka na A1/A2) | tylko gdy fitness ocalałej ≥ fitness usuwanej − `δ` | `δ = 0`: wartości i priorytety identyczne w chwili redukcji; `δ > 0`: odchylenie ograniczone przez `δ` | `δ` to „próg redukcji” z planu pracy |
| A4 | Przycinanie na dowodach z pamięci | reguły bezczynne: na próbce przejść z pamięci nie są ani najlepszym wiarygodnym antycypatorem, ani nośnikiem maksimum fitness, ani nie biorą udziału w przejściu bez wiarygodnej antycypacji; dopiero po k kolejnych audytach | tylko empiryczna | jedyny kandydat na niewiarygodną większość; wymaga strażnika i pomiaru odtwarzania |
| A5 | Progi Choczaja (wersja poprawna) | punkt odniesienia | brak | strojenie ręczne |
| A6 | Twardy limit z eksmisją najmniej użytecznych | nadmiar ponad N | twarda granica pamięci | stratne; opcjonalne |
| A7 | Dźwignie wbudowane w ACS2: `do_ga`, małe `u_max` | zapobiegają nadmiernej specjalizacji | — | tylko komparator; recenzent zapyta „po co kompaktować, skoro można włączyć GA” |

**Rekomendacja:** rdzeń bezpieczny A2+A3; kandydat ambitny A4; A1, A5, A7 jako punkty odniesienia; A6 tylko jeśli potrzebna twarda granica.

### B. Jak nie zepsuć (strażnik)

- Niezmienniki z konstrukcji (A2, A3 przy `δ = 0`) sprawdzane testami własności, bez kosztu w czasie działania.
- Dla operatorów stratnych audyt przed i po na próbce z pamięci: pokrycie przejść wiarygodną antycypacją, zmiana `Q` i akcji zachłannej, przesunięcie rozkładu akceptacji VCP. Przekroczenie progu oznacza odrzucenie. Cofnięcie jest darmowe, bo `compact_population` zwraca nową populację.
- Pokrycie liczymy także dla przejść bez zmiany stanu (ruch w ścianę). Miara wiedzy, wartości i wybór akcji ich nie widzą, więc utrata takich reguł byłaby niewidoczna, a reguły wracałyby przez pokrywanie.
- Strażnik algorytmu widzi wyłącznie pamięć agenta. Pełny zbiór przejść środowiska służy tylko pomiarowi.
- Przenoszenie predykcji nagrody na subsumera (pomysł z CRACS) zmienia wartości w innych stanach, które subsumer pokrywa. Traktuję je tylko jako wariant badawczy obok strażnika `δ`.

### C. Kiedy i jak mocno (wyzwalacz i sterowanie)

| | Wariant | Rola |
|---|---|---|
| C0 | nigdy / raz na końcu | punkty odniesienia |
| C1 | stały okres co N interakcji | punkt odniesienia z planu pracy |
| C2 | wzrost populacji: redukcja, gdy populacja urosła ρ razy względem rozmiaru po poprzedniej redukcji | prosty wariant adaptacyjny, jeden parametr |
| C3 | koszt–zysk: przewidywane zaoszczędzone testy dopasowania > koszt przebiegu; obie strony liczone deterministycznie tym samym licznikiem | bez progów strojonych per środowisko |
| C4 | sprzężenie zwrotne: mały uzysk → rzadziej; odtwarzanie usuniętych reguł lub odrzucenie przez strażnika → łagodniej (`δ` w dół, dojrzałość w górę, k w górę); w przeciwnym razie śmielej | właściwa „adaptacyjność” |
| C5 | sygnał fazy uczenia (tempo tworzenia reguł, pokrycie pamięci) | wyłącznie modulacja, nigdy gwarancja |

**Rekomendacja:** C2 jako najprostszy wariant, C3+C4 jako cel, C1 do porównania. Hipoteza robocza: przebieg kosztuje tyle co kilkaset interakcji, więc o częstości decyduje głównie ryzyko, a adaptacja powinna sterować przede wszystkim **siłą** redukcji. Etap 3 to rozstrzygnie, zanim zainwestujemy w wyrafinowany wyzwalacz.

### D. Wiele głowic

- Każda głowica kompaktuje się sama, na granicy epizodu. W Rust priorytety są liczone na świeżo przy każdym losowaniu, więc po redukcji nie ma nieaktualnych priorytetów.
- Warianty do porównania: każda głowica według własnego zegara albo wszystkie naraz.
- Hipoteza specyficzna dla VCP: stratna redukcja jednej głowicy podnosi wariancję tam, gdzie zmieniła wartości; VCP losuje takie próbki rzadziej, więc naprawa zwalnia. To argument za strażnikiem wartości.
- Ryzyko zbieżności: ten sam deterministyczny operator na wszystkich głowicach może je upodobnić (wariancja → 0, VCP → losowanie jednorodne). Mierzymy niezgodność głowic w czasie.

## Rekomendowana ścieżka: lejek z bramkami

**Etap 0 — definicje (bez kodu).** Zamrozić miary sukcesu i marginesy, podział labiryntów na strojenie i ocenę, listę ramion, budżet. Doczytać prace o kompaktowaniu w trakcie uczenia w XCS/UCS, żeby poprawnie ustawić nowość: subsumpcja w zbiorze akcji, Dixon i in. 2002 oraz praca o kompaktowaniu online dla UCS, którą znalazłem tylko z tytułu („An Online Rule Set Compaction Strategy for the sUpervised Classifier System UCS”) i której treści jeszcze nie zweryfikowałem.

**Etap 1 — operatory i audyt jako czyste funkcje.** A1–A5 oraz porównanie przed/po, bez wpływu na jakąkolwiek ścieżkę uczenia. Reguły rozstrzygane od najogólniejszych, żeby tolerancja `δ` się nie kumulowała w łańcuchach. Testy własności: równość wiedzy, neutralność wartości, idempotencja, determinizm.

**Etap 2 — „kompaktowanie w cieniu”.** Przebiegi bazowe czterech rodzin agentów oraz ramiona A7; w każdym punkcie kontrolnym każdy operator działa na *kopii* populacji i zapisujemy: ile usuwa, skład populacji (wiarygodne, oznaczone, doświadczenie, liczba wyspecyfikowanych atrybutów celu, nośniki wartości) oraz natychmiastową szkodę.
- Wynik: krzywe „ile da się usunąć w czasie i czym”, front kompresja–szkoda oraz rozbicie kosztu dopasowania na wybór akcji, uczenie i zapytania priorytetu (profil z kroku 3 planu linii).
- **Bramka:** jeśli A2+A3 usuwają mało, ciężar przechodzi na A4; dalej idą najwyżej trzy operatory.

**Etap 3 — pojedyncza interwencja.** Jedna redukcja w chwili wczesnej, środkowej albo późnej, potem zwykłe uczenie. Mierzymy odrost populacji, odsetek odtworzonych reguł, powrót wiedzy i sukcesu, zaoszczędzone testy dopasowania.
- Prefiks odtwarzamy deterministycznie z ziarna, bez nowego kodeka. Po redukcji trajektorie się rozchodzą (wybór akcji zużywa losowość zależnie od liczby reguł), więc porównanie jest statystyczne, na co najmniej 10 ziarnach.
- **Bramka:** trwała oszczędność netto bez opóźnienia uczenia; mapa momentów, w których redukcja szkodzi.

**Etap 4 — sterowanie adaptacyjne.** C1, C2, C3+C4 na zbiorze strojenia, z analizą wrażliwości.
- **Bramka:** wariant adaptacyjny nie gorszy od najlepszego stałego okresu, bez strojenia per labirynt.

**Etap 5 — efekty wielogłowicowe i równoległość.** Koordynacja głowic, rozkład priorytetów, niezgodność; osobno pomiar równoległego liczenia głowic, żeby odróżnić zysk algorytmu od zysku implementacji.

**Etap 6 — porównanie końcowe.** Zamrożona konfiguracja, sześć labiryntów, 20 ziaren, równa liczba interakcji sumowana po głowicach; ACS2, ACS2ER, ACS2HER, ACS2VCP (K=2, K=4) i ich kontrole bez priorytetów. Wyniki negatywne też raportujemy.

## Miejsce w kodzie

- `acs2-core/src/compaction/` — nowe moduły obok nietkniętego `cracs.rs`: kryteria subsumpcji, strażnik wartości, przycinanie na dowodach, audyt, spis populacji, polityka wyzwalania. Rozszerzyć `CompactableRule` o fitness i stan znacznika. Wzorzec zwracania decyzji: `CompactionOutcome`.
- `acs2-agents/src/{acs2vcp,acs2her,acs2er,acs2}/` — warianty kompaktujące z własnym typem parametrów (wzorzec `ConsistencyConfiguration`, `ReplayConfiguration`); potrzebny mały szew do podmiany populacji między epizodami (`acs2vcp::Agent`, `episodic_replay::EpisodicReplay`).
- `acs2-measure/src/runner.rs`, `agents/` — nowe `AgentKind` i nazwy przebiegów; diagnostyka kompaktowania jako opcjonalna metoda `GoalAgent` z domyślnym `None`, żeby istniejące wiersze się nie zmieniły.
- `experiments/rule-compaction/` — rejestracja i analiza (wzorzec `experiments/value-consistency/`, uruchamianie przez `experiments/grid/`).
- `docs/RULE_COMPACTION.md`, `docs/ARCHITECTURE.md` — stan linii i opis mechanizmu.
- Bez zmian: `Configuration`, `ReplaySample`, kodek punktów kontrolnych, domyślne zachowania. Losowość kompaktowania dostaje własny strumień (zajęte: 1–10 i 12).

Do ponownego użycia: `anticipation_fraction` (`support/knowledge.rs`), liczniki `MatchCounters` (`support/measurement.rs`), `Agent::variance` i diagnostyka niezgodności (`acs2-measure/src/agents/vcp/`), `TrajectoryStore` i `Sampler` (`acs2-trajectory`), przypięte wiersze `fixtures/vcp_pair_regression.json`.

## Protokół i budżet

- Stanowisko główne: `acs2-measure --protocol maze-pairs-v1` (cele jako współrzędne, bloki treningowy/walidacyjny/testowy, 2 mln interakcji, Woods14 4 mln).
- Strojenie: MazeMA, Woods14, Maze4, blok walidacyjny, ziarna 42–46. Ocena: sześć labiryntów, blok testowy, ziarna 42–61; dowodem głównym są Maze5, Maze6, Maze7, nietknięte przy strojeniu.
- Test dodatkowy w reżimie Zemły (cel jako percepcja, krótki horyzont, Maze6/MazeF3/MazeB), bo tam eksplozja jest najgwałtowniejsza.
- Szacunek z udokumentowanych kosztów: etap 2 do 6 h, etap 3 do 20 h, etap 4 do 25 h, etap 5 do 10 h, etap 6 do 140 h. Razem ok. 200 z 500 CPU-h. Wersja oszczędna etapu 6 (10 ziaren albo bez kontroli jednorodnych): ok. 70 h.
- Każdy etap na klastrze zaczyna się od sprawdzenia grantu i pilotażu kosztu na ziarnie 42. Nic nie liczymy na laptopie.

## Miary sukcesu (propozycja do zamrożenia w etapie 0)

- Koszt: testy dopasowania w treningu plus praca kompaktowania w tej samej jednostce (deterministyczne, niezależne od sprzętu); osobno czas ścienny, pamięć i pole pod krzywą rozmiaru populacji.
- Jakość: wiedza, sukces na parach wstrzymanych (średnia z punktów kontrolnych i punkt końcowy), diagnostyka wartości.
- „Pomaga” znaczy: co najmniej 30% mniej testów dopasowania przy wiedzy i sukcesie nie gorszych o więcej niż 0,01 względem tego samego agenta bez kompaktowania, na labiryntach nieużytych do strojenia.

## Ryzyka

- Większość nadmiaru to reguły niewiarygodne, które usuwa dopiero operator stratny. Etap 2 pokaże to tanio.
- Usunięte reguły odrastają i zysk znika. Mierzy to etap 3.
- W tym repo ACS2VCP z priorytetami wypada słabiej od własnej kontroli jednorodnej (wynik udokumentowany w `experiments/README.md`). Tezy trzeba formułować względem tego samego agenta bez kompaktowania i trzymać kontrole jako ramiona.
- Odczyt ze środka przebiegu wzięty za wynik końcowy. Wnioski tylko z zarejestrowanych punktów i pełnych horyzontów.

## Weryfikacja

- Testy własności na losowych populacjach (wzorzec `random_populations_keep_their_knowledge` w `cracs.rs`): równość wiedzy, równość `MaxFitnessBootstrap` przed i po dla wszystkich percepcji i akcji, drugi przebieg nic nie usuwa.
- Kompaktowanie wyłączone daje wiersze bit w bit zgodne z istniejącymi presetami (`acs2-measure/tests/acs2vcp.rs`, przypięta fikstura).
- Tożsamość liczników `formations = environment_steps + 3·full + 2·values_only + priority_value_queries` nadal zachodzi; praca kompaktowania liczona osobno.
- Bramki z `CLAUDE.md` uruchamiane na WCSS: testy, P9 bajt w bajt w kolumnach 1–7, clippy bez nowych ostrzeżeń.
- Test całościowy: pilotaż na MazeMA, ziarno 42 — wiersze zawierają diagnostykę każdej redukcji (rozmiar przed i po, powody decyzji, zmiana na próbce z pamięci), a populacja i liczba testów spadają przy wiedzy w marginesie.

## Założenia domyślne — popraw, jeśli chcesz inaczej

1. Celem głównym jest koszt obliczeń przy nie gorszej jakości, a nie twardy limit pamięci.
2. Stanowiskiem głównym jest istniejący protokół par start–cel; reżim Zemły to test dodatkowy.
3. VCP w temperaturze 24, jak w ukończonym badaniu w repo, zawsze z kontrolą jednorodną.
4. Budżet do ok. 200 CPU-h, każdy etap klastrowy za Twoją zgodą.
5. GA i małe `u_max` występują tylko jako komparatory, nie jako część proponowanej metody.
6. Pierwszy krok po akceptacji: etapy 0 i 1 (definicje, operatory i audyt bez zmiany zachowania czegokolwiek).
