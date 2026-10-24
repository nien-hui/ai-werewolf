/* AI 狼人殺 frontend. Vanilla JS (ES2017), no build step.
 * Talks to werewolf/server.py over a WebSocket. All dynamic text is set via
 * textContent (see h()); never innerHTML with model/user text. Icons and avatars are
 * static <symbol>s in index.html, referenced with <use> (see icon() / avatar()). */
(function () {
  'use strict';

  // ---------------------------------------------------------------- constants
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const ROLE_ZH = { werewolf: '狼人', seer: '預言家', witch: '女巫', villager: '村民' };
  const ROLE_TEAM = { werewolf: 'wolf', seer: 'good', witch: 'good', villager: 'good' };
  const TEAM_ZH = { wolf: '狼人陣營', good: '好人陣營', draw: '平局' };
  const ROLE_ICON = { werewolf: 'wolf', seer: 'seer', witch: 'potion', villager: 'wheat' };
  const ROLE_DESC = {
    werewolf: '夜晚和狼隊友商量擊殺一人，白天偽裝成好人；殺光神職或殺光村民就獲勝。',
    seer: '每晚查驗一名玩家是狼人還是好人，用資訊帶領好人找出狼人。',
    witch: '有一瓶解藥、一瓶毒藥，各限用一次；不能自救，同一晚不能兩瓶都用。',
    villager: '沒有特殊能力，靠聽發言和投票把狼人找出來。',
  };
  const THOUGHT_ACTION_ZH = {
    speech: '白天發言', last_words: '遺言', wolf_chat: '狼隊夜聊', wolf_kill: '選擇擊殺',
    witch: '女巫用藥', seer_check: '查驗', exile_vote: '放逐投票',
  };
  const SPEAK_TITLE = {
    discussion: { title: '輪到你發言', icon: 'chat' },
    last_words: { title: '發表遺言', icon: 'candle' },
    wolf_chat: { title: '狼隊夜聊', icon: 'wolf', chip: '只有狼隊友看得到' },
  };
  const CHOOSE_TITLE = {
    wolf_kill: { title: '選擇今晚的擊殺目標', icon: 'claw' },
    seer_check: { title: '選擇要查驗的玩家', icon: 'seer' },
    exile_vote: { title: '放逐投票', icon: 'ballot' },
  };
  const CONFIRM_VERB = { wolf_kill: '確認擊殺', seer_check: '確認查驗', exile_vote: '確認投給' };
  const PHASE_META = {
    night: { icon: 'moon', sub: '夜晚行動' },
    day: { icon: 'sun', sub: '白天討論' },
    end: { icon: 'flag', sub: '' },
  };
  const SEAT_VARIANTS = 8; // avatar palettes / masks in style.css + index.html
  const STORAGE_KEY = 'whoIsHuman.gameId';
  const PASS_KEY = 'whoIsHuman.key'; // the server's password (a ?key= link sets it)
  // the server closes the socket with this when the password is missing or wrong
  const AUTH_CLOSE = 4401;
  const MAX_CHARS = 200;
  // AI speeches appear one character at a time, TYPE_MS each, but never take longer than TYPE_MAX_MS.
  // The server waits for exactly that (werewolf/config.py TYPE_S_PER_CHAR / TYPE_MAX_S), plus 2 s.
  const TYPE_MS = 55;
  const TYPE_MAX_MS = 6000;
  const TYPED_KINDS = ['speech', 'last_words', 'wolf_chat'];
  const NEAR_BOTTOM_PX = 80;
  const BASE_TITLE = document.title;
  const PARAMS = new URLSearchParams(location.search);

  // -------------------------------------------------------------------- state
  const S = {
    ws: null,
    connected: false,
    retry: 0,
    retryTimer: 0,
    mode: 'start', // start | game | end
    awaiting: null, // 'new' | 'resume' until game_started arrives
    resumeInFlight: false,
    needKey: false, // the server wants a password: no connecting until the player gives one
    badKey: false, // ... because the one this browser had was wrong
    gameId: null,
    you: null, // {seat, role, role_zh}
    numPlayers: 6,
    players: [], // [{seat, alive, is_you, tag}]
    day: 0,
    phase: 'setup',
    phaseKey: '', // last phase|day drawn in the phase banner (drives the phase transition)
    phaseQuiet: false, // resumed game: draw the current phase without the transition
    splashTimer: 0,
    youOpen: false, // phones: role description expanded
    youOut: false, // the role card shows 已出局
    seen: new Set(), // rendered event ids
    prompt: null,
    promptKey: '',
    panel: null, // {root, refresh(enabled), setNote(text), pick?, focus?, cls}
    panelState: 'idle', // idle | active | sent
    selection: undefined, // choose prompt: seat number, null = 棄票, undefined = nothing
    speakingSeat: null,
    typing: null, // the AI speech being typed out: {seat, finish()}
    stick: true, // log is scrolled to the bottom
    forceScroll: false,
  };

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const isNum = (v) => typeof v === 'number' && isFinite(v);

  /** Create an element. props: class, text, on<event>, or attributes. */
  function h(tag, props) {
    const node = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach((k) => {
        const v = props[k];
        if (v === null || v === undefined || v === false) return;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = String(v);
        else if (k.slice(0, 2) === 'on' && typeof v === 'function') node.addEventListener(k.slice(2), v);
        else node.setAttribute(k, v === true ? '' : String(v));
      });
    }
    for (let i = 2; i < arguments.length; i++) add(node, arguments[i]);
    return node;
  }

  function add(node, c) {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) {
      c.forEach((x) => add(node, x));
      return;
    }
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }

  function str(v) {
    return v === null || v === undefined ? '' : String(v);
  }

  /** Text as DOM text nodes, with every 「N號」 in a nowrap span so it never breaks across lines.
   *  Built only from text nodes / textContent — never parsed as HTML. */
  function richText(text) {
    const t = str(text);
    const frag = document.createDocumentFragment();
    const re = /\d+號/g;
    let last = 0;
    let m;
    while ((m = re.exec(t))) {
      if (m.index > last) frag.appendChild(document.createTextNode(t.slice(last, m.index)));
      frag.appendChild(h('span', { class: 'nw', text: m[0] }));
      last = m.index + m[0].length;
    }
    if (last < t.length) frag.appendChild(document.createTextNode(t.slice(last)));
    return frag;
  }

  function svgUse(id, cls) {
    const u = document.createElementNS(SVG_NS, 'use');
    u.setAttribute('href', '#' + id);
    if (cls) u.setAttribute('class', cls);
    return u;
  }

  /** Decorative inline SVG icon from the sprite in index.html. */
  function icon(name, cls) {
    const s = document.createElementNS(SVG_NS, 'svg');
    s.setAttribute('class', 'i' + (cls ? ' ' + cls : ''));
    s.setAttribute('aria-hidden', 'true');
    s.setAttribute('focusable', 'false');
    s.appendChild(svgUse('i-' + name));
    return s;
  }

  /** 1..SEAT_VARIANTS: picks the seat's palette (.sc-N) and mask (#av-mN); 0 = unknown seat. */
  function seatVariant(seat) {
    return isNum(seat) ? ((Math.max(1, Math.floor(seat)) - 1) % SEAT_VARIANTS) + 1 : 0;
  }

  /** Per-seat hooded-figure avatar. extra classes: 'sm' (number coin only), 'lg', 'nonum'. */
  function avatar(seat, me, extra) {
    const v = seatVariant(seat);
    const ex = ' ' + str(extra) + ' ';
    const node = h('span', { class: 'avatar sc-' + v + (me ? ' me' : '') + (extra ? ' ' + extra : ''), 'aria-hidden': 'true' });
    if (ex.indexOf(' sm ') >= 0) {
      node.textContent = isNum(seat) ? String(seat) : '?';
      return node;
    }
    const s = document.createElementNS(SVG_NS, 'svg');
    s.setAttribute('class', 'av-svg');
    s.setAttribute('viewBox', '0 0 48 48');
    s.setAttribute('focusable', 'false');
    s.appendChild(svgUse('av-hood', 'av-hood'));
    if (v) s.appendChild(svgUse('av-m' + v, 'av-mask'));
    node.appendChild(s);
    if (ex.indexOf(' nonum ') < 0) node.appendChild(h('span', { class: 'av-num', text: isNum(seat) ? seat : '?' }));
    return node;
  }

  /** "3號", plus role in the review and a "you" marker. */
  function seatLabel(seat, ctx) {
    if (!isNum(seat)) return '？';
    let s = seat + '號';
    if (ctx && ctx.review && ctx.roles[seat]) s += '・' + (ROLE_ZH[ctx.roles[seat]] || ctx.roles[seat]);
    if (ctx && ctx.mySeat === seat) s += ctx.review ? '（真人・你）' : '（你）';
    return s;
  }

  function seatChip(seat, ctx, cls) {
    return h('span', { class: 'seat-chip' + (cls ? ' ' + cls : '') },
      avatar(seat, ctx && ctx.mySeat === seat, 'sm'), h('span', null, richText(seatLabel(seat, ctx))));
  }

  function visSeat(ev) {
    const v = str(ev.visibility);
    return v.indexOf('seat:') === 0 ? Number(v.slice(5)) : null;
  }

  function phaseLabel(day, phase) {
    if (phase === 'setup') return '開局';
    if (phase === 'night') return '第' + day + '夜';
    if (phase === 'day') return '第' + day + '天';
    if (phase === 'end') return '終局';
    return str(phase) || '其他';
  }

  function tagClass(tag) {
    const t = str(tag);
    if (t.indexOf('好人') >= 0) return 'good';
    if (t.indexOf('狼') >= 0) return 'wolf';
    return '';
  }

  /** Seat tag pill. "查驗：好人" keeps its full text; the prefix is visually hidden (the eye icon shows it). */
  function tagNode(tag) {
    const t = str(tag);
    const kind = tagClass(t);
    const m = /^(查驗)\s*[：:]\s*(.+)$/.exec(t);
    const node = h('span', { class: 'tag ' + kind, title: t }, icon(m ? 'eye' : kind === 'wolf' ? 'wolf' : 'sparkle'));
    if (m) add(node, [h('span', { class: 'tag-pre', text: m[1] + '：' }), h('span', { text: m[2] })]);
    else add(node, h('span', { text: t }));
    return node;
  }

  function playerBySeat(seat) {
    for (let i = 0; i < S.players.length; i++) if (S.players[i].seat === seat) return S.players[i];
    return null;
  }

  function mySeat() {
    if (S.you && isNum(S.you.seat)) return S.you.seat;
    for (let i = 0; i < S.players.length; i++) if (S.players[i].is_you) return S.players[i].seat;
    return null;
  }

  /** Remove a leading "3號：" the speaker typed (the bubble already shows who spoke). Only that
   *  exact prefix, the same rule the AIs' log uses, so "3號預言家：…" stays whole. */
  function stripSpeaker(text, seat) {
    const t = str(text);
    if (!isNum(seat)) return t;
    return t.replace(new RegExp('^\\s*' + seat + '號[：:]\\s*'), '') || t;
  }

  function storeGameId(id) {
    try {
      if (id) localStorage.setItem(STORAGE_KEY, id);
      else localStorage.removeItem(STORAGE_KEY);
    } catch (e) { /* storage unavailable */ }
  }

  function storedGameId() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null;
    }
  }

  function storeKey(key) {
    try {
      if (key) localStorage.setItem(PASS_KEY, key);
      else localStorage.removeItem(PASS_KEY);
    } catch (e) { /* storage unavailable */ }
  }

  function storedKey() {
    try {
      return localStorage.getItem(PASS_KEY);
    } catch (e) {
      return null;
    }
  }

  function reducedMotion() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  /** Set a button's visible label without dropping its icon. */
  function setLabel(btn, text) {
    const l = btn.querySelector('.btn-label');
    if (l) l.textContent = text;
    else btn.textContent = text;
  }

  function toast(text, kind) {
    const box = $('toasts');
    while (box.childNodes.length >= 4) box.removeChild(box.firstChild);
    const t = h('div', { class: 'toast' + (kind ? ' ' + kind : ''), role: 'status' },
      icon(kind === 'info' ? 'sparkle' : 'alert'), h('span', { text: text }));
    box.appendChild(t);
    setTimeout(() => {
      t.classList.add('out');
      setTimeout(() => t.remove(), 350);
    }, 4500);
  }

  // ------------------------------------------------------------- connection
  function wsUrl() {
    const override = PARAMS.get('ws');
    if (override) return override;
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    const key = storedKey();
    return proto + (location.host || 'localhost:8000') + '/ws' + (key ? '?key=' + encodeURIComponent(key) : '');
  }

  function connect() {
    clearTimeout(S.retryTimer);
    let ws;
    try {
      ws = new WebSocket(wsUrl());
    } catch (e) {
      scheduleReconnect();
      return;
    }
    S.ws = ws;
    ws.onopen = () => {
      if (S.ws !== ws) return;
      S.connected = true;
      S.retry = 0;
      updateConn();
      onOpen();
    };
    ws.onmessage = (e) => {
      if (S.ws !== ws) return;
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch (err) {
        console.warn('non-JSON message', e.data);
        return;
      }
      try {
        handle(msg);
      } catch (err) {
        console.error('message handler failed', err, msg);
      }
    };
    ws.onclose = (e) => {
      if (S.ws !== ws) return;
      S.ws = null;
      S.connected = false;
      if (e && e.code === AUTH_CLOSE) { // missing or wrong password: ask for it, then connect again
        if (storedKey()) { // it was wrong: forget it
          storeKey(null);
          S.badKey = true;
        }
        S.needKey = true;
        S.awaiting = null;
        if (S.mode !== 'start') showScreen('start');
        updateConn();
        const input = $('key-input');
        if (input && wantsAutofocus()) input.focus();
        return;
      }
      updateConn();
      scheduleReconnect();
    };
    ws.onerror = () => { /* onclose follows */ };
  }

  function scheduleReconnect() {
    clearTimeout(S.retryTimer);
    const delay = Math.min(10000, 600 * Math.pow(2, S.retry)) + Math.random() * 400;
    S.retry = Math.min(S.retry + 1, 10);
    S.retryTimer = setTimeout(connect, delay);
  }

  function reconnectNow() {
    if (S.ws || S.needKey) return;
    S.retry = 0;
    connect();
  }

  function send(msg) {
    if (!S.ws || S.ws.readyState !== WebSocket.OPEN) return false;
    try {
      S.ws.send(JSON.stringify(msg));
      return true;
    } catch (e) {
      return false;
    }
  }

  function newGameMsg() {
    const msg = { type: 'new_game' };
    const seed = PARAMS.get('seed');
    if (seed !== null && /^-?\d+$/.test(seed)) msg.seed = Number(seed);
    return msg;
  }

  function sendResume(id) {
    S.resumeInFlight = true;
    send({ type: 'resume_game', game_id: id });
  }

  function onOpen() {
    if (S.mode === 'game' && S.gameId) {
      sendResume(S.gameId);
    } else if (S.awaiting === 'new') {
      send(newGameMsg());
    } else if (S.awaiting === 'resume') {
      const id = storedGameId();
      if (id) sendResume(id);
      else S.awaiting = null;
    }
    renderStart();
  }

  function updateConn() {
    const dot = $('conn-dot');
    dot.classList.toggle('on', S.connected);
    dot.title = S.connected ? '已連線' : '連線中斷';
    dot.setAttribute('aria-label', dot.title);
    $('conn-banner').hidden = S.connected || S.mode !== 'game';
    if (S.mode === 'start') renderStart();
  }

  // --------------------------------------------------------------- dispatch
  function handle(msg) {
    if (!msg || typeof msg !== 'object') return;
    switch (msg.type) {
      case 'game_started': onGameStarted(msg); break;
      case 'players': onPlayers(msg); break;
      case 'event': onEvent(msg.event); break;
      case 'status': setStatus(msg.text); break;
      case 'prompt': onPrompt(msg.prompt); break;
      case 'error': onError(msg.message); break;
      case 'game_end': onGameEnd(msg.review); break;
      default: console.debug('unknown message type', msg.type);
    }
  }

  function onGameStarted(msg) {
    const id = str(msg.game_id);
    const same = S.mode === 'game' && S.gameId === id;
    const resumed = S.awaiting === 'resume';
    S.awaiting = null;
    S.resumeInFlight = false;
    if (!same) resetGame();
    if (resumed) S.phaseQuiet = true;
    S.gameId = id;
    if (msg.you && typeof msg.you === 'object') S.you = msg.you;
    if (isNum(msg.num_players)) S.numPlayers = msg.num_players;
    storeGameId(id);
    if (S.mode !== 'game') showScreen('game');
    renderYou();
    renderPhase();
    if (S.panelState === 'idle') renderIdle();
    else renderSeats();
  }

  function onPlayers(msg) {
    if (Array.isArray(msg.players)) S.players = msg.players.filter((p) => p && isNum(p.seat));
    if (isNum(msg.day)) S.day = msg.day;
    if (typeof msg.phase === 'string') S.phase = msg.phase;
    renderPhase();
    if (S.you && youOut() !== S.youOut) renderYou();
    if (S.panelState === 'idle') renderIdle();
    else renderSeats();
  }

  function onEvent(ev) {
    if (!ev || typeof ev !== 'object') return;
    if (isNum(ev.id)) {
      if (S.seen.has(ev.id)) return;
      S.seen.add(ev.id);
    }
    finishTyping(); // anything new shows the speech being typed in full first
    const node = renderEvent(ev, logCtx());
    appendLog(node);
    if (TYPED_KINDS.indexOf(ev.kind) >= 0 && isNum(ev.seat) && ev.seat !== mySeat()) {
      const b = node.querySelector('.bubble');
      if (b) typeOut(b, stripSpeaker(ev.text, ev.seat), ev.seat);
    }
    // A new event after we answered means the server accepted the action.
    if (S.panelState === 'sent') renderIdle();
  }

  function onPrompt(p) {
    if (!p || typeof p !== 'object') return;
    setStatus('');
    const key = JSON.stringify(p);
    if (S.panel && S.prompt && key === S.promptKey) {
      // Re-sent after reconnect: keep the draft, just make sure it is usable.
      if (S.panelState === 'sent') {
        S.panel.setNote('');
        setPanelEnabled(true);
      }
      return;
    }
    S.prompt = p;
    S.promptKey = key;
    S.selection = undefined;
    renderPrompt();
  }

  function onError(message) {
    const text = str(message) || '發生錯誤';
    if (S.awaiting === 'resume' || (S.resumeInFlight && S.mode === 'game')) {
      // Resume failed: the game is gone (e.g. server restarted).
      S.awaiting = null;
      S.resumeInFlight = false;
      S.gameId = null;
      storeGameId(null);
      resetGame();
      showScreen('start');
      toast('無法繼續上一局：' + text);
      return;
    }
    if (S.awaiting === 'new') {
      S.awaiting = null;
      renderStart();
      renderEndButtons();
    }
    toast(text);
    if (S.panel && S.panelState === 'sent') {
      S.panel.setNote('');
      setPanelEnabled(true);
      setStatus('');
    }
  }

  function onGameEnd(review) {
    finishTyping();
    S.awaiting = null;
    S.resumeInFlight = false;
    storeGameId(null);
    setStatus('');
    S.prompt = null;
    S.promptKey = '';
    S.panel = null;
    S.panelState = 'idle';
    document.title = BASE_TITLE;
    try {
      renderEnd(review && typeof review === 'object' ? review : {});
    } catch (e) {
      console.error('renderEnd failed', e);
    }
    showScreen('end');
    renderEndButtons();
  }

  // ---------------------------------------------------------------- screens
  function showScreen(name) {
    S.mode = name;
    ['start', 'game', 'end'].forEach((n) => {
      $('screen-' + n).hidden = n !== name;
    });
    applyTheme();
    updateConn();
    if (name !== 'game') window.scrollTo(0, 0);
  }

  function applyTheme() {
    // The game opens in the dark (setup precedes night 1), so avoid a day flash.
    const night = S.mode === 'start' || (S.mode === 'game' && (S.phase === 'night' || S.phase === 'setup'));
    document.body.classList.toggle('night', night);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', night ? '#16114a' : '#eef6ff');
  }

  function resetGame() {
    finishTyping();
    S.seen = new Set();
    S.players = [];
    S.day = 0;
    S.phase = 'setup';
    S.phaseKey = '';
    S.phaseQuiet = false;
    S.youOut = false;
    S.prompt = null;
    S.promptKey = '';
    S.panel = null;
    S.panelState = 'idle';
    S.selection = undefined;
    S.speakingSeat = null;
    S.stick = true;
    $('log').textContent = '';
    $('jump-new').hidden = true;
    $('status-text').textContent = '';
    $('status').hidden = true;
    $('hud-alive').hidden = true;
  }

  function startNew() {
    S.awaiting = 'new';
    S.resumeInFlight = false;
    renderStart();
    renderEndButtons();
    if (S.connected) send(newGameMsg());
    else reconnectNow();
  }

  function resumeSaved() {
    const id = storedGameId();
    if (!id) {
      renderStart();
      return;
    }
    S.awaiting = 'resume';
    renderStart();
    if (S.connected) sendResume(id);
    else reconnectNow();
  }

  function renderStart() {
    const saved = storedGameId();
    const btnNew = $('btn-new');
    const btnRes = $('btn-resume');
    btnRes.hidden = !saved;
    btnNew.disabled = !!S.awaiting || S.needKey;
    btnRes.disabled = !!S.awaiting || S.needKey;
    $('key-form').hidden = !S.needKey;
    btnNew.classList.toggle('busy', S.awaiting === 'new');
    btnRes.classList.toggle('busy', S.awaiting === 'resume');
    setLabel(btnNew, S.awaiting === 'new' ? '建立遊戲中…' : '開始新遊戲');
    setLabel(btnRes, S.awaiting === 'resume' ? '載入中…' : '繼續上一局');
    let hint = '';
    if (S.needKey) hint = S.badKey ? '密碼不對，請再輸入一次。' : '';
    else if (!S.connected) hint = '正在連線伺服器…';
    else if (S.awaiting === 'new') hint = '正在分配座位與身份…';
    else if (saved) hint = '偵測到一局尚未結束的遊戲，可以繼續上一局。';
    $('start-hint').textContent = hint;
  }

  function renderEndButtons() {
    const b = $('btn-again');
    b.disabled = S.awaiting === 'new';
    b.classList.toggle('busy', S.awaiting === 'new');
    setLabel(b, S.awaiting === 'new' ? '建立遊戲中…' : '再來一局');
  }

  /** Decorative line-up of the six hooded players on the title screen, in a shallow arc. */
  function buildLineup() {
    const box = $('hero-lineup');
    if (!box) return;
    box.textContent = '';
    const n = 6;
    const mid = (n - 1) / 2;
    for (let s = 1; s <= n; s++) {
      const seat = h('span', { class: 'lu-seat' }, avatar(s, false, 'lg nonum'), h('span', { class: 'lu-num', text: s }));
      seat.style.setProperty('--arc', String(Math.pow((s - 1 - mid) / mid, 2)));
      box.appendChild(seat);
    }
  }

  // --------------------------------------------------------- game: header/side
  function renderPhase() {
    const badge = $('phase-badge');
    let text = '準備中';
    if (S.phase === 'night') text = '第' + S.day + '夜';
    else if (S.phase === 'day') text = '第' + S.day + '天';
    else if (S.phase === 'end') text = '遊戲結束';
    const known = ['night', 'day', 'end'].indexOf(S.phase) >= 0;
    const key = S.phase + '|' + S.day;
    if (key !== S.phaseKey) {
      // only a change seen live gets the transition, not the first draw after a load or resume
      const live = S.phaseKey !== '' && !S.phaseQuiet;
      if (known) S.phaseQuiet = false;
      const meta = known ? PHASE_META[S.phase] : { icon: 'hourglass', sub: '' };
      badge.textContent = '';
      add(badge, [
        h('span', { class: 'pb-icon' }, icon(meta.icon)),
        h('span', { class: 'pb-text' }, h('b', { text: text }), meta.sub ? h('small', { text: meta.sub }) : null),
      ]);
      badge.className = 'phase-badge ' + (known ? S.phase : '');
      if (live && S.mode === 'game' && (S.phase === 'night' || S.phase === 'day')) {
        if (!reducedMotion()) {
          void badge.offsetWidth; // restart the pop animation
          badge.classList.add('pop');
        }
        phaseSplash(S.phase, text);
      }
      S.phaseKey = key;
    }
    renderHud();
    applyTheme();
  }

  /** Brief "第N夜 / 第N天" splash over the log when the phase changes (skipped for reduced motion). */
  function phaseSplash(phase, text) {
    const el = $('phase-splash');
    if (!el || reducedMotion()) return;
    el.textContent = '';
    el.className = 'phase-splash ' + phase;
    add(el, h('div', { class: 'ps-card' },
      h('span', { class: 'ps-icon' }, icon(phase === 'night' ? 'moon' : 'sun')),
      h('span', { class: 'ps-title', text: text }),
      h('span', { class: 'ps-sub', text: phase === 'night' ? '天黑請閉眼' : '天亮了' })));
    void el.offsetWidth; // restart the animation
    el.classList.add('show');
    clearTimeout(S.splashTimer);
    S.splashTimer = setTimeout(() => el.classList.remove('show'), 1150);
  }

  function renderHud() {
    const el = $('hud-alive');
    if (!S.players.length) {
      el.hidden = true;
      return;
    }
    const alive = S.players.filter((p) => p.alive).length;
    el.hidden = false;
    el.textContent = '';
    el.title = '存活 ' + alive + ' / ' + S.players.length;
    add(el, [icon('users'), h('span', { class: 'hud-label', text: '存活' }), h('span', { class: 'hud-count' }, h('b', { text: alive }), '/' + S.players.length)]);
  }

  function youOut() {
    const me = playerBySeat(mySeat());
    return !!me && !me.alive;
  }

  function renderYou() {
    const card = $('you-card');
    card.textContent = '';
    const you = S.you;
    if (!you) return;
    const role = str(you.role);
    const team = ROLE_TEAM[role];
    S.youOut = youOut();
    card.className = 'you-card ' + (team || '') + (S.youOut ? ' out' : '') + (S.youOpen ? ' open' : '');
    // phones fold the role description behind this button (hidden on desktop)
    const more = h('button', {
      class: 'you-more', type: 'button', title: '身份說明', 'aria-label': '身份說明',
      'aria-controls': 'you-desc', 'aria-expanded': S.youOpen ? 'true' : 'false',
    }, icon('info'));
    more.addEventListener('click', () => {
      S.youOpen = !S.youOpen;
      card.classList.toggle('open', S.youOpen);
      more.setAttribute('aria-expanded', S.youOpen ? 'true' : 'false');
    });
    add(card, [
      h('div', { class: 'you-emblem' }, icon(ROLE_ICON[role] || 'mask')),
      h('div', { class: 'you-main' },
        h('div', { class: 'you-top' },
          h('div', { class: 'you-title' }, '你是 ' + str(you.seat) + '號・',
            h('span', { class: 'role', text: str(you.role_zh) || ROLE_ZH[role] || role })),
          team ? h('span', { class: 'you-team' }, icon(team === 'wolf' ? 'wolf' : 'shield'), TEAM_ZH[team]) : null,
          S.youOut ? h('span', { class: 'you-out' }, icon('skull'), '已出局') : null),
        h('div', { class: 'you-desc', id: 'you-desc', text: ROLE_DESC[role] || '' })),
      more,
    ]);
  }

  function renderSeats() {
    const box = $('seats');
    // the seat cards are rebuilt: keep keyboard focus on the same seat
    const active = document.activeElement;
    const focusSeat = active && box.contains(active) ? active.getAttribute('data-seat') : null;
    box.textContent = '';
    const me = mySeat();
    let list = S.players;
    if (!list.length) {
      list = [];
      for (let s = 1; s <= S.numPlayers; s++) list.push({ seat: s, alive: true, is_you: s === me, tag: null });
    }
    const pickable = (S.panelState === 'active' && S.prompt && S.prompt.type === 'choose' && Array.isArray(S.prompt.options))
      ? S.prompt.options : null;
    box.classList.toggle('pick-kill', !!pickable && S.prompt.action === 'wolf_kill');
    list.forEach((p) => {
      const isMe = !!p.is_you || p.seat === me;
      const canPick = !!pickable && pickable.indexOf(p.seat) >= 0;
      const speaking = S.speakingSeat === p.seat;
      const cls = ['seat'];
      if (isMe) cls.push('me');
      if (!p.alive) cls.push('dead');
      if (speaking) cls.push('speaking');
      if (canPick) cls.push('selectable');
      if (canPick && S.selection === p.seat) cls.push('selected');
      if (p.tag) cls.push('has-tag');
      let state;
      if (!p.alive) state = h('span', { class: 'seat-state dead-text' }, icon('skull'), '出局');
      else if (speaking) state = h('span', { class: 'seat-state speaking-text' }, icon('wave'), '發言中');
      else state = h('span', { class: 'seat-state alive-text' }, h('i', { class: 'live-dot' }), '存活');
      const node = h('div', {
        class: cls.join(' '), 'data-seat': p.seat,
        title: p.seat + '號' + (isMe ? '（你）' : '') + (p.alive ? '' : '・已出局'),
      },
        h('div', { class: 'seat-av' },
          avatar(p.seat, isMe, 'lg nonum'),
          !p.alive ? h('span', { class: 'dead-mark' }, icon('skull')) : null,
          speaking ? h('span', { class: 'speak-mark' }, icon('wave')) : null,
          canPick ? h('span', { class: 'pick-mark' }, icon('check')) : null,
          isMe ? h('span', { class: 'me-tag', text: '你' }) : null),
        h('div', { class: 'seat-name' }, h('b', { text: p.seat }), h('small', { text: '號' })),
        h('div', { class: 'seat-sub' }, state, p.tag ? tagNode(p.tag) : null));
      if (canPick) {
        node.setAttribute('role', 'button');
        node.setAttribute('tabindex', '0');
        node.setAttribute('aria-pressed', S.selection === p.seat ? 'true' : 'false');
        const pick = () => {
          if (S.panel && S.panel.pick) S.panel.pick(p.seat);
        };
        node.addEventListener('click', pick);
        node.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            pick();
          }
        });
      }
      box.appendChild(node);
    });
    if (focusSeat !== null) {
      const n = box.querySelector('[data-seat="' + focusSeat + '"]');
      if (n && n.tabIndex >= 0) {
        try {
          n.focus({ preventScroll: true });
        } catch (e) {
          n.focus();
        }
      }
    }
  }

  // ------------------------------------------------------------------- log
  function isNearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
  }

  function scrollToBottom() {
    const log = $('log');
    log.scrollTop = log.scrollHeight;
    S.stick = true;
    $('jump-new').hidden = true;
  }

  /** Run fn (which may change layout) and keep the log pinned if it was. */
  function keepBottom(fn) {
    const stick = S.stick;
    fn();
    if (stick && S.mode === 'game') scrollToBottom();
  }

  function appendLog(node) {
    const log = $('log');
    const stick = S.forceScroll || S.stick;
    S.forceScroll = false;
    log.appendChild(node);
    if (stick) scrollToBottom();
    else $('jump-new').hidden = false;
  }

  /** Seat named by a "N號 正在…" status, else the seat whose speech is being typed out. */
  function speakingNow() {
    const m = /^\s*(\d+)\s*號\s*正在/.exec($('status-text').textContent);
    return m ? Number(m[1]) : S.typing ? S.typing.seat : null;
  }

  function markSpeaking() {
    const speaking = speakingNow();
    if (speaking !== S.speakingSeat) {
      S.speakingSeat = speaking;
      renderSeats();
    }
  }

  function setStatus(text) {
    const t = typeof text === 'string' ? text : '';
    const named = /^\s*(\d+)\s*號\s*正在/.test(t);
    keepBottom(() => {
      $('status-text').textContent = t;
      $('status').hidden = !t;
      $('status').classList.toggle('speaking', named);
    });
    markSpeaking();
  }

  /** Type `text` into the speech bubble `el` one character at a time (the seat card shows 發言中
   *  meanwhile). Screen readers get the whole text at once; reduced motion shows it right away. */
  function typeOut(el, text, seat) {
    const chars = Array.from(text);
    if (reducedMotion() || chars.length < 2) return;
    const shown = h('span', { 'aria-hidden': 'true' });
    el.textContent = '';
    add(el, [shown, h('span', { class: 'sr-only', text: text })]);
    el.classList.add('typing');
    const step = Math.min(TYPE_MS, TYPE_MAX_MS / chars.length);
    const log = $('log');
    let n = 0;
    let timer = 0;
    const job = { seat: seat, finish: finish };
    function finish() {
      clearTimeout(timer);
      if (S.typing === job) S.typing = null;
      el.textContent = '';
      el.appendChild(richText(text));
      el.classList.remove('typing');
      markSpeaking();
    }
    function tick() {
      n = Math.min(chars.length, n + 1);
      shown.textContent = chars.slice(0, n).join('');
      if (S.stick && S.mode === 'game') log.scrollTop = log.scrollHeight;
      if (n >= chars.length) finish();
      else timer = setTimeout(tick, step);
    }
    S.typing = job;
    markSpeaking();
    timer = setTimeout(tick, step);
  }

  function finishTyping() {
    if (S.typing) S.typing.finish();
  }

  function logCtx() {
    return { review: false, mySeat: mySeat(), roles: {} };
  }

  // -------------------------------------------------------- event rendering
  const RENDER = {
    system: renderSystem,
    role_assign: (ev, ctx) => privateLine(ev, ctx, 'mask'),
    wolf_chat: (ev, ctx) => bubble(ev, ctx, 'wolf'),
    wolf_pick: (ev, ctx) => wolfLine(ev, ctx, false),
    wolf_kill: (ev, ctx) => wolfLine(ev, ctx, true),
    witch_info: (ev, ctx) => privateLine(ev, ctx, 'potion'),
    witch_action: (ev, ctx) => privateLine(ev, ctx, 'potion'),
    seer_result: (ev, ctx) => privateLine(ev, ctx, 'seer'),
    death: renderDeath,
    speech: (ev, ctx) => bubble(ev, ctx, 'speech'),
    last_words: (ev, ctx) => bubble(ev, ctx, 'last_words'),
    vote_result: renderVote,
    game_over: renderGameOver,
    reveal: renderReveal,
    thought: renderThought,
  };

  function renderEvent(ev, ctx) {
    const fn = Object.prototype.hasOwnProperty.call(RENDER, ev.kind) ? RENDER[ev.kind] : null;
    if (fn) {
      try {
        return fn(ev, ctx);
      } catch (e) {
        console.warn('render failed, using fallback', ev, e);
      }
    }
    return renderGeneric(ev, ctx);
  }

  function renderGeneric(ev, ctx) {
    const v = str(ev.visibility);
    if (v.indexOf('seat:') === 0) return privateLine(ev, ctx, '');
    if (v === 'wolves') return wolfLine(ev, ctx, false);
    if (v === 'god') return renderThought(ev, ctx);
    return h('div', { class: 'ev-generic' }, richText(str(ev.text) || ('（' + str(ev.kind) + '）')));
  }

  function renderSystem(ev) {
    const ph = ev.phase;
    if (ph === 'night' || ph === 'day') {
      return h('div', { class: 'ev-system divider ' + ph },
        h('span', { class: 'div-line', 'aria-hidden': 'true' }),
        h('span', { class: 'div-body' }, icon(ph === 'night' ? 'moon' : 'sun'), h('span', null, richText(ev.text))),
        h('span', { class: 'div-line', 'aria-hidden': 'true' }));
    }
    return h('div', { class: 'ev-system' }, icon('sparkle'), h('span', null, richText(ev.text)));
  }

  function bubble(ev, ctx, variant) {
    const seat = ev.seat;
    const mine = isNum(seat) && seat === ctx.mySeat;
    let label = null;
    if (variant === 'last_words') label = h('span', { class: 'label-lw' }, icon('candle'), '遺言');
    if (variant === 'wolf') label = h('span', { class: 'label-wolf' }, icon('wolf'), '狼隊頻道');
    return h('div', { class: 'msg ' + variant + (mine ? ' mine' : '') },
      avatar(seat, mine),
      h('div', { class: 'msg-body' },
        h('div', { class: 'msg-meta' }, h('span', { class: 'who', text: seatLabel(seat, ctx) }), label),
        h('div', { class: 'bubble' }, richText(stripSpeaker(ev.text, seat)))));
  }

  function privateLine(ev, ctx, iconName) {
    const owner = visSeat(ev);
    const lock = ctx.review ? (isNum(owner) ? owner : str(ev.seat)) + '號私密' : '只有你看得到';
    return h('div', { class: 'ev-private' },
      h('span', { class: 'ev-ico' }, icon(iconName || 'lock')),
      h('div', { class: 'ev-main' },
        h('span', { class: 'lock' }, icon('lock'), lock),
        h('span', { class: 'ev-text' }, richText(ev.text))));
  }

  function wolfLine(ev, ctx, kill) {
    let text = str(ev.text);
    const d = ev.data || {};
    if (!text && isNum(d.target)) {
      text = kill ? '狼隊決定擊殺 ' + d.target + '號' : seatLabel(ev.seat, ctx) + ' 想殺 ' + d.target + '號';
    }
    return h('div', { class: 'ev-wolf' + (kill ? ' kill' : '') },
      h('span', { class: 'ev-ico' }, icon(kill ? 'claw' : 'target')),
      h('div', { class: 'ev-main' },
        h('span', { class: 'chan' }, icon('wolf'), '狼隊頻道'),
        h('span', { class: 'ev-text' }, richText(text))));
  }

  function cardHead(iconName, title) {
    return h('div', { class: 'card-head' },
      h('span', { class: 'card-ico' }, icon(iconName)),
      title instanceof Node ? title : h('div', { class: 'card-title', text: title }));
  }

  function renderDeath(ev, ctx) {
    const seats = Array.isArray(ev.data && ev.data.seats) ? ev.data.seats.filter(isNum) : [];
    const peace = seats.length === 0;
    return h('div', { class: 'ev-card ' + (peace ? 'peace' : 'death') },
      cardHead(peace ? 'sunrise' : 'skull', h('div', { class: 'card-title' },
        h('span', { text: peace ? '平安夜' : '死亡公告' }),
        peace ? null : h('span', { class: 'chips' }, seats.map((s) => seatChip(s, ctx))))),
      h('div', { class: 'card-text' }, richText(ev.text)));
  }

  /** Vote count bars, most votes first (same data the old "票數：…" line showed). */
  function tallyBars(tally, exiled) {
    if (!tally || typeof tally !== 'object') return null;
    const parts = Object.keys(tally)
      .map((k) => ({ k: k, n: Number(tally[k]) || 0 }))
      .filter((x) => x.n > 0)
      .sort((a, b) => b.n - a.n || Number(a.k) - Number(b.k));
    if (!parts.length) return null;
    const max = Math.max.apply(null, parts.map((x) => x.n));
    return h('div', { class: 'tally', role: 'list', 'aria-label': '票數' }, parts.map((x) => {
      const seat = /^\d+$/.test(x.k) ? Number(x.k) : null;
      const bar = h('i');
      bar.style.width = Math.max(8, Math.round((x.n / max) * 100)) + '%';
      return h('div', { class: 'tally-row' + (seat !== null && seat === exiled ? ' hit' : '') + (x.n === max ? ' top' : ''), role: 'listitem' },
        seat !== null ? avatar(seat, false, 'sm') : h('span', { class: 'avatar sm skip-coin', 'aria-hidden': 'true' }, icon('ban')),
        h('span', { class: 'tally-who', text: seat !== null ? seat + '號' : '棄票' }),
        h('span', { class: 'tally-bar' }, bar),
        h('span', { class: 'tally-n', text: x.n + ' 票' }));
    }));
  }

  function renderVote(ev, ctx) {
    const d = ev.data || {};
    const votes = Array.isArray(d.votes) ? d.votes : null;
    if (!votes) return cardFallback('ballot', '投票結果', ev);
    const exiled = isNum(d.exiled) ? d.exiled : null;
    const body = h('tbody');
    votes.forEach((v) => {
      const target = v && isNum(v.target) ? v.target : null;
      body.appendChild(h('tr', { class: target !== null && target === exiled ? 'hit' : null },
        h('td', null, seatChip(v && v.voter, ctx)),
        h('td', { class: 'arrow' }, icon('arrow-down', 'arrow-r')),
        h('td', null, target === null ? h('span', { class: 'skip' }, icon('ban'), '棄票') : seatChip(target, ctx))));
    });
    let result;
    if (exiled !== null) result = h('div', { class: 'card-result exiled' }, icon('gavel'), h('span', null, seatLabel(exiled, ctx), ' 被放逐出局'));
    else if (votes.some((v) => v && isNum(v.target))) result = h('div', { class: 'card-result' }, icon('scales'), h('span', { text: '平票，無人出局' }));
    else result = h('div', { class: 'card-result' }, icon('ban'), h('span', { text: '全員棄票，無人出局' }));
    return h('div', { class: 'ev-card vote' },
      cardHead('ballot', (isNum(ev.day) ? '第' + ev.day + '天 ' : '') + '放逐投票結果'),
      h('div', { class: 'vote-body' },
        h('table', { class: 'mini-table' },
          h('thead', null, h('tr', null, h('th', { text: '投票者' }), h('th', { 'aria-hidden': 'true' }), h('th', { text: '投給' }))),
          body),
        tallyBars(d.tally, exiled)),
      result);
  }

  function renderGameOver(ev) {
    const d = ev.data || {};
    const w = d.winner;
    let title = '遊戲結束';
    if (w === 'draw') title += '：平局';
    else if (TEAM_ZH[w]) title += '：' + TEAM_ZH[w] + '獲勝';
    return h('div', { class: 'ev-card over' },
      cardHead('flag', title),
      h('div', { class: 'card-text' }, richText(str(d.reason) || str(ev.text))));
  }

  function renderReveal(ev, ctx) {
    const d = ev.data || {};
    const roles = Array.isArray(d.roles) ? d.roles.filter((r) => r && isNum(r.seat)) : null;
    if (!roles) return cardFallback('mask', '身份公開', ev);
    return h('div', { class: 'ev-card reveal' },
      cardHead('mask', '身份公開'),
      h('div', { class: 'role-chips' }, roles.map((r) =>
        h('span', { class: 'role-chip' + (r.role === 'werewolf' ? ' wolf' : '') },
          avatar(r.seat, r.seat === ctx.mySeat, 'sm'),
          icon(ROLE_ICON[r.role] || 'mask'),
          h('span', { text: r.seat + '號 ' + (ROLE_ZH[r.role] || str(r.role)) }),
          r.is_human ? h('span', { class: 'human-mark', text: '真人' }) : null))));
  }

  function renderThought(ev, ctx) {
    const d = ev.data || {};
    const act = THOUGHT_ACTION_ZH[d.action] || str(d.action);
    return h('div', { class: 'ev-card thought' },
      h('div', { class: 'thought-head' },
        h('span', { class: 'vis-tag god' }, icon('thought'), '內心'),
        avatar(ev.seat, false, 'sm'),
        h('b', { text: seatLabel(ev.seat, ctx) }),
        act ? h('span', { text: '（' + act + '）' }) : null),
      h('div', { class: 'thought-text' }, richText(ev.text)));
  }

  function cardFallback(iconName, title, ev) {
    return h('div', { class: 'ev-card' },
      cardHead(iconName, title),
      h('div', { class: 'card-text' }, richText(ev.text)));
  }

  // ------------------------------------------------------------ action panel
  function renderIdle() {
    S.prompt = null;
    S.promptKey = '';
    S.panel = null;
    S.panelState = 'idle';
    S.selection = undefined;
    document.title = BASE_TITLE;
    keepBottom(() => {
      const root = $('action');
      const info = idleInfo();
      root.className = 'action idle' + (info.cls ? ' ' + info.cls : '');
      root.textContent = '';
      root.appendChild(h('div', { class: 'idle-row' }, h('span', { class: 'idle-ico' }, icon(info.icon)), h('span', { text: info.text })));
    });
    renderSeats();
  }

  function idleInfo() {
    const me = playerBySeat(mySeat());
    if (me && !me.alive) return { icon: 'ghost', text: '你已出局，靜靜觀戰中…', cls: 'spectator' };
    if (S.phase === 'night') return { icon: 'moon', text: '夜深了，靜待天亮…' };
    if (S.phase === 'day') return { icon: 'eye', text: '仔細聽其他玩家的發言…' };
    if (S.phase === 'end') return { icon: 'flag', text: '遊戲結束…' };
    return { icon: 'hourglass', text: '遊戲準備中…' };
  }

  function renderPrompt() {
    const p = S.prompt;
    keepBottom(() => {
      const root = $('action');
      root.textContent = '';
      let panel;
      if (p.type === 'speak') panel = buildSpeak(p);
      else if (p.type === 'choose') panel = buildChoose(p);
      else if (p.type === 'witch') panel = buildWitch(p);
      else panel = buildUnknown(p);
      S.panel = panel;
      S.panelState = 'active';
      root.className = 'action active' + (panel.cls ? ' ' + panel.cls : '');
      root.appendChild(panel.root);
      panel.refresh(true);
    });
    document.title = '🔔 輪到你了｜' + BASE_TITLE;
    renderSeats();
    if (S.panel && S.panel.focus) S.panel.focus();
  }

  function setPanelEnabled(enabled) {
    if (!S.panel) return;
    S.panelState = enabled ? 'active' : 'sent';
    $('action').classList.toggle('sent', !enabled);
    S.panel.refresh(enabled);
    renderSeats();
  }

  function sendAction(value, note) {
    if (S.panelState !== 'active' || !S.panel) return;
    if (!send({ type: 'action', value: value })) {
      toast('目前沒有連線，請稍候再試');
      return;
    }
    setPanelEnabled(false);
    S.panel.setNote((note || '已送出') + '，等待其他玩家…');
    S.forceScroll = true;
    setStatus('處理中…');
  }

  /** Panel frame: icon tile, title, prompt text, the 「輪到你了」 badge and the sent note. */
  function panelShell(title, promptText, iconName, chip) {
    const noteText = h('span');
    const note = h('div', { class: 'sent-note', role: 'status', hidden: true }, icon('check'), noteText);
    const root = h('div', { class: 'act' },
      h('div', { class: 'act-head' },
        h('span', { class: 'act-icon' }, icon(iconName || 'bolt')),
        h('div', { class: 'act-heading' },
          h('div', { class: 'act-title' }, h('span', { text: title }),
            chip ? h('span', { class: 'act-chip' }, icon('lock'), chip) : null,
            h('span', { class: 'turn-badge' }, h('i', { class: 'turn-dot' }), '輪到你了')),
          promptText ? h('div', { class: 'act-prompt' }, richText(promptText)) : null)));
    return {
      root: root,
      note: note,
      setNote: (t) => {
        noteText.textContent = t || '';
        note.hidden = !t;
      },
    };
  }

  function wantsAutofocus() {
    return !(window.matchMedia && window.matchMedia('(max-width: 760px), (pointer: coarse)').matches);
  }

  function buildSpeak(p) {
    const kind = p.kind;
    const meta = SPEAK_TITLE[kind] || SPEAK_TITLE.discussion;
    const shell = panelShell(meta.title, str(p.prompt), meta.icon, meta.chip);
    const ta = h('textarea', {
      class: 'speech',
      rows: 3,
      maxlength: MAX_CHARS,
      placeholder: kind === 'last_words' ? '留下你的遺言…' : kind === 'wolf_chat' ? '說出你今晚想刀誰、白天怎麼打…' : '輸入你的發言…',
      'aria-label': '發言內容',
    });
    const count = h('span', { class: 'char-count' });
    const passBtn = h('button', {
      class: 'btn ghost small pass', type: 'button', text: '過', title: '跳過這次發言（會替你說一句簡短的場面話）',
    });
    const sendBtn = h('button', { class: 'btn primary send', type: 'button' }, icon('send'), h('span', { text: '送出' }));
    let enabled = true;
    let composing = false;
    let passArmed = 0;

    const length = () => Array.from(ta.value).length;
    function refresh(en) {
      if (en !== undefined) enabled = en;
      const n = length();
      count.textContent = n + ' / ' + MAX_CHARS;
      count.classList.toggle('over', n > MAX_CHARS);
      ta.disabled = !enabled;
      passBtn.disabled = !enabled;
      sendBtn.disabled = !enabled || !ta.value.trim() || n > MAX_CHARS;
    }
    function submit() {
      // speeches are one line for everyone (the server joins lines too)
      const text = ta.value.replace(/\s*[\r\n]+\s*/g, ' ').trim();
      if (!enabled || !text || Array.from(text).length > MAX_CHARS) return;
      sendAction({ text: text }, '發言已送出');
    }
    function resetPass() {
      clearTimeout(passArmed);
      passArmed = 0;
      passBtn.textContent = '過';
      passBtn.classList.remove('armed');
    }

    ta.addEventListener('input', () => refresh());
    ta.addEventListener('compositionstart', () => { composing = true; });
    ta.addEventListener('compositionend', () => { setTimeout(() => { composing = false; }, 0); });
    ta.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      // Enter that confirms an IME candidate must not send (zh-TW users type with IME).
      if (e.isComposing || e.keyCode === 229 || composing) return;
      e.preventDefault(); // no line breaks: Shift+Enter does nothing
      if (!e.shiftKey) submit();
    });
    sendBtn.addEventListener('click', submit);
    passBtn.addEventListener('click', () => {
      if (!passArmed) {
        passBtn.textContent = '確定要過？';
        passBtn.classList.add('armed');
        passArmed = setTimeout(resetPass, 3000);
        return;
      }
      resetPass();
      sendAction({ text: '' }, '你選擇了過（已替你說一句場面話）');
    });

    add(shell.root, [
      h('div', { class: 'compose' }, ta),
      h('div', { class: 'act-foot' },
        count,
        h('span', { class: 'kbd-hint' }, h('kbd', { text: 'Enter' }), ' 送出'),
        h('span', { class: 'grow' }),
        passBtn,
        sendBtn),
      shell.note,
    ]);
    return {
      root: shell.root,
      cls: kind === 'wolf_chat' ? 'wolfy' : kind === 'last_words' ? 'farewell' : '',
      refresh: refresh,
      setNote: shell.setNote,
      focus: () => {
        if (wantsAutofocus()) {
          try {
            ta.focus({ preventScroll: true });
          } catch (e) {
            ta.focus();
          }
        }
      },
    };
  }

  /** Seat option tile for choose / poison panels; its visible text always contains 「N號」.
   *  sub: optional second line (text or a tag pill). */
  function seatOption(seat, sub, cls) {
    return h('button', { class: 'seat-btn' + (cls ? ' ' + cls : ''), type: 'button', 'aria-pressed': 'false' },
      avatar(seat, false, 'nonum'),
      h('span', { class: 'sb-text' },
        h('span', { class: 'sb-label', text: seat + '號' }),
        sub ? h('span', { class: 'sub' }, sub) : null),
      h('span', { class: 'sb-check', 'aria-hidden': 'true' }, icon('check')));
  }

  function buildChoose(p) {
    const action = p.action;
    const meta = CHOOSE_TITLE[action] || { title: '請選擇', icon: 'target' };
    const shell = panelShell(meta.title, str(p.prompt), meta.icon);
    const options = Array.isArray(p.options) ? p.options.filter(isNum) : [];
    const allowSkip = !!p.allow_skip;
    const buttons = new Map();
    let enabled = true;

    const row = h('div', { class: 'act-row options' });
    options.forEach((seat) => {
      const pl = playerBySeat(seat);
      const sub = pl && !pl.alive ? '出局' : pl && pl.tag ? tagNode(pl.tag) : '';
      const b = seatOption(seat, sub, '');
      b.addEventListener('click', () => pick(seat));
      buttons.set(seat, b);
      row.appendChild(b);
    });
    let skipBtn = null;
    if (allowSkip) {
      skipBtn = h('button', { class: 'seat-btn skip-btn', type: 'button', 'aria-pressed': 'false' },
        h('span', { class: 'skip-ico', 'aria-hidden': 'true' }, icon('ban')),
        h('span', { class: 'sb-text' }, h('span', { class: 'sb-label', text: '棄票' })),
        h('span', { class: 'sb-check', 'aria-hidden': 'true' }, icon('check')));
      skipBtn.addEventListener('click', () => pick(null));
      row.appendChild(skipBtn);
    }
    if (!options.length && !allowSkip) row.appendChild(h('span', { class: 'act-label', text: '沒有可選擇的對象' }));
    const confirmBtn = h('button', { class: 'btn confirm ' + (action === 'wolf_kill' ? 'danger' : 'primary'), type: 'button' });

    function pick(seat) {
      if (!enabled) return;
      if (seat === null ? !allowSkip : !buttons.has(seat)) return;
      S.selection = seat;
      refresh();
      renderSeats();
    }
    function refresh(en) {
      if (en !== undefined) enabled = en;
      const sel = S.selection;
      buttons.forEach((b, seat) => {
        b.disabled = !enabled;
        b.classList.toggle('selected', sel === seat);
        b.setAttribute('aria-pressed', sel === seat ? 'true' : 'false');
      });
      if (skipBtn) {
        skipBtn.disabled = !enabled;
        skipBtn.classList.toggle('selected', sel === null);
        skipBtn.setAttribute('aria-pressed', sel === null ? 'true' : 'false');
      }
      const has = sel === null ? allowSkip : isNum(sel) && buttons.has(sel);
      confirmBtn.disabled = !enabled || !has;
      confirmBtn.textContent = !has ? '請先選擇' : sel === null ? '確認棄票' : (CONFIRM_VERB[action] || '確認') + ' ' + sel + '號';
    }
    confirmBtn.addEventListener('click', () => {
      const sel = S.selection;
      if (sel === null && allowSkip) {
        sendAction({ target: null }, '你選擇了棄票');
      } else if (isNum(sel) && buttons.has(sel)) {
        sendAction({ target: sel }, '已選擇 ' + sel + '號');
      }
    });

    add(shell.root, [
      row,
      h('div', { class: 'act-foot' },
        h('span', { class: 'foot-hint' }, icon('target'), '也可以直接點座位卡'),
        h('span', { class: 'grow' }), confirmBtn),
      shell.note,
    ]);
    return {
      root: shell.root,
      cls: action === 'wolf_kill' ? 'wolfy' : '',
      refresh: refresh,
      setNote: shell.setNote,
      pick: pick,
    };
  }

  function buildWitch(p) {
    // the rows below show the night's death and every choice; p.prompt would only repeat them
    const shell = panelShell('女巫請行動', '每晚最多只能使用一瓶藥。', 'potion');
    const me = mySeat();
    const killed = isNum(p.killed) ? p.killed : null;
    const poisonOpts = Array.isArray(p.poison_options) ? p.poison_options.filter(isNum) : [];
    const canSave = !!p.can_save && killed !== null && killed !== me;
    const poisonBtns = new Map();
    let poisonSel = null;
    let enabled = true;
    let saveBtn = null;

    if (killed !== null) {
      shell.root.appendChild(h('div', { class: 'act-info danger' }, icon('drop'),
        h('span', { text: '今晚 ' + killed + '號' + (killed === me ? '（你自己）' : '') + ' 被狼人殺害' })));
    } else {
      shell.root.appendChild(h('div', { class: 'act-info' }, icon('eye'), h('span', { text: '你已用過解藥，無法得知今晚的死訊。' })));
    }

    const saveBody = h('div', { class: 'potion-body' });
    const saveRow = h('div', { class: 'potion save' },
      h('div', { class: 'potion-label' }, h('span', { class: 'p-ico' }, icon('heart')), h('span', { text: '解藥' })), saveBody);
    if (canSave) {
      saveBtn = h('button', { class: 'btn good', type: 'button' }, icon('heart'), h('span', { text: '使用解藥救 ' + killed + '號' }));
      saveBtn.addEventListener('click', () => sendAction({ action: 'save', target: null }, '已使用解藥救 ' + killed + '號'));
      saveBody.appendChild(saveBtn);
    } else if (killed !== null && killed === me) {
      saveBody.appendChild(h('span', { class: 'act-label', text: '女巫不能自救。' }));
    } else if (killed !== null) {
      saveBody.appendChild(h('span', { class: 'act-label', text: '這晚無法使用解藥。' }));
    } else {
      saveBody.appendChild(h('span', { class: 'act-label', text: '已用完。' }));
    }

    const poisonBody = h('div', { class: 'potion-body' });
    const poisonRow = h('div', { class: 'potion poison' },
      h('div', { class: 'potion-label' }, h('span', { class: 'p-ico' }, icon('poison')), h('span', { text: '毒藥' })), poisonBody);
    const poisonConfirm = h('button', { class: 'btn danger', type: 'button' });
    if (poisonOpts.length) {
      const opts = h('div', { class: 'act-row options compact' });
      poisonOpts.forEach((seat) => {
        const b = seatOption(seat, '', 'poison');
        b.addEventListener('click', () => {
          if (!enabled) return;
          poisonSel = poisonSel === seat ? null : seat;
          refresh();
        });
        poisonBtns.set(seat, b);
        opts.appendChild(b);
      });
      poisonBody.appendChild(opts);
      poisonConfirm.addEventListener('click', () => {
        if (poisonSel !== null) sendAction({ action: 'poison', target: poisonSel }, '已對 ' + poisonSel + '號 使用毒藥');
      });
    } else {
      poisonBody.appendChild(h('span', { class: 'act-label', text: '已用完。' }));
    }

    const noneBtn = h('button', { class: 'btn ghost', type: 'button', text: '這晚不使用' });
    noneBtn.addEventListener('click', () => sendAction({ action: 'none', target: null }, '這晚不使用藥水'));

    function refresh(en) {
      if (en !== undefined) enabled = en;
      if (saveBtn) saveBtn.disabled = !enabled;
      poisonBtns.forEach((b, seat) => {
        b.disabled = !enabled;
        b.classList.toggle('selected', poisonSel === seat);
        b.setAttribute('aria-pressed', poisonSel === seat ? 'true' : 'false');
      });
      poisonConfirm.disabled = !enabled || poisonSel === null;
      poisonConfirm.textContent = poisonSel === null ? '選擇要毒的人' : '使用毒藥毒 ' + poisonSel + '號';
      noneBtn.disabled = !enabled;
    }

    // like the other panels, the committing buttons sit in the footer
    add(shell.root, [
      h('div', { class: 'potions' }, saveRow, poisonRow),
      h('div', { class: 'act-foot' }, h('span', { class: 'grow' }), poisonOpts.length ? poisonConfirm : null, noneBtn),
      shell.note,
    ]);
    return { root: shell.root, cls: '', refresh: refresh, setNote: shell.setNote };
  }

  function buildUnknown(p) {
    const shell = panelShell('需要你行動', str(p.prompt), 'alert');
    shell.root.appendChild(h('div', { class: 'act-info', text: '無法辨識的行動類型：' + str(p.type) + '。請重新整理頁面。' }));
    shell.root.appendChild(shell.note);
    return { root: shell.root, cls: '', refresh: () => {}, setNote: shell.setNote };
  }

  // -------------------------------------------------------------- end screen
  function renderEnd(rv) {
    const players = (Array.isArray(rv.players) ? rv.players : [])
      .filter((p) => p && isNum(p.seat))
      .sort((a, b) => a.seat - b.seat);
    const humanP = players.filter((p) => p.is_human)[0];
    const humanSeat = isNum(rv.human_seat) ? rv.human_seat : humanP ? humanP.seat : mySeat();
    const me = players.filter((p) => p.seat === humanSeat)[0] || (S.you ? { seat: S.you.seat, role: S.you.role } : {});
    const myTeam = ROLE_TEAM[me.role];
    const winner = rv.winner;
    const roles = {};
    const personas = {};
    players.forEach((p) => {
      roles[p.seat] = p.role;
      personas[p.seat] = p.is_human ? '真人（你）' : str(p.persona_name) || 'AI';
    });
    const ctx = { review: true, mySeat: humanSeat, roles: roles };
    const events = (Array.isArray(rv.events) ? rv.events : []).filter((e) => e && typeof e === 'object');
    const lastPhase = events.filter((e) => e.phase === 'night' || e.phase === 'day').pop();

    // hero
    const hero = $('result-hero');
    hero.textContent = '';
    let outcome = '遊戲結束';
    let cls = '';
    if (winner === 'draw') {
      outcome = '平局';
      cls = 'draw';
    } else if (winner && myTeam) {
      cls = winner === myTeam ? 'win' : 'lose';
      outcome = cls === 'win' ? '你的陣營獲勝！' : '你的陣營落敗了';
    }
    hero.className = 'result-hero ' + cls;
    const emblem = { win: 'crown', lose: 'shield-broken', draw: 'scales' }[cls] || 'flag';
    const kicker = '終局' + (lastPhase ? '・' + phaseLabel(lastPhase.day, lastPhase.phase) : '');
    const teamText = winner === 'draw' ? '平局' : TEAM_ZH[winner] ? TEAM_ZH[winner] + '獲勝' : '';
    const meText = isNum(me.seat) && me.role ? '你是 ' + me.seat + '號 ' + (ROLE_ZH[me.role] || me.role) : '';
    add(hero, [
      h('div', { class: 'hero-rays', 'aria-hidden': 'true' }),
      h('div', { class: 'hero-emblem' }, icon(emblem)),
      h('div', { class: 'hero-kicker', text: kicker }),
      h('h1', { class: 'hero-outcome', text: outcome }),
      (teamText || meText) ? h('div', { class: 'hero-team' },
        teamText ? h('span', { class: 'hero-pill ' + (winner === 'wolf' ? 'wolf' : winner === 'good' ? 'good' : '') },
          icon(winner === 'wolf' ? 'wolf' : winner === 'good' ? 'shield' : 'scales'), teamText) : null,
        meText ? h('span', { class: 'hero-pill me' }, isNum(me.seat) ? avatar(me.seat, true, 'sm') : null,
          icon(ROLE_ICON[me.role] || 'mask'), meText) : null) : null,
      rv.win_reason ? h('div', { class: 'hero-sub', text: str(rv.win_reason) }) : null,
    ]);

    // roles
    const grid = $('roles-grid');
    grid.textContent = '';
    players.forEach((p) => {
      grid.appendChild(h('div', { class: 'role-card' + (p.role === 'werewolf' ? ' wolf' : '') + (p.is_human ? ' human' : '') + (p.alive ? '' : ' dead') },
        h('div', { class: 'rc-av' },
          avatar(p.seat, !!p.is_human, 'lg nonum'),
          p.is_human ? h('span', { class: 'me-tag', text: '你' }) : null),
        h('div', { class: 'rc-body' },
          h('div', { class: 'rc-top' },
            h('span', { class: 'rc-seat', text: p.seat + '號' }),
            h('span', { class: 'rc-role' }, icon(ROLE_ICON[p.role] || 'mask'), str(p.role_zh) || ROLE_ZH[p.role] || str(p.role))),
          h('div', { class: 'rc-sub' },
            h('span', { class: 'rc-persona' }, icon(p.is_human ? 'user' : 'bot'), personas[p.seat]),
            h('span', { class: 'rc-state' + (p.alive ? '' : ' dead') }, p.alive ? h('i', { class: 'live-dot' }) : icon('skull'), p.alive ? '存活' : '出局')))));
    });

    // god-view timeline grouped by day/phase
    const tl = $('timeline');
    tl.textContent = '';
    let group = null;
    let key = null;
    events.forEach((ev) => {
      const k = str(ev.day) + '|' + str(ev.phase);
      if (k !== key) {
        key = k;
        const ph = str(ev.phase);
        group = h('div', { class: 'tl-group ' + ph },
          h('div', { class: 'tl-head' },
            icon(ph === 'night' ? 'moon' : ph === 'day' ? 'sun' : ph === 'end' ? 'flag' : 'sparkle'),
            phaseLabel(ev.day, ev.phase)));
        tl.appendChild(group);
      }
      group.appendChild(renderEvent(ev, ctx));
    });
    if (!events.length) tl.appendChild(h('div', { class: 'ev-generic', text: '沒有可顯示的紀錄' }));
    tl.classList.toggle('hide-thoughts', !$('toggle-thoughts').checked);
    $('review').open = false;
  }

  // ------------------------------------------------------------------- init
  function init() {
    $('btn-new').addEventListener('click', startNew);
    $('btn-resume').addEventListener('click', resumeSaved);
    $('btn-again').addEventListener('click', startNew);
    $('key-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const key = $('key-input').value;
      if (!key) return;
      storeKey(key);
      $('key-input').value = '';
      S.needKey = false;
      S.badKey = false;
      renderStart();
      reconnectNow();
    });
    const linkKey = PARAMS.get('key');
    if (linkKey) storeKey(linkKey);
    if (linkKey !== null) { // keep the password out of history and screenshots
      PARAMS.delete('key');
      const query = PARAMS.toString();
      try {
        history.replaceState(null, '', location.pathname + (query ? '?' + query : ''));
      } catch (e) { /* not allowed here (e.g. file://) */ }
    }
    $('btn-home').addEventListener('click', () => {
      showScreen('start');
      renderStart();
    });
    $('toggle-thoughts').addEventListener('change', (e) => {
      $('timeline').classList.toggle('hide-thoughts', !e.target.checked);
    });
    $('jump-new').addEventListener('click', scrollToBottom);
    $('phase-badge').addEventListener('animationend', (e) => e.currentTarget.classList.remove('pop'));

    const log = $('log');
    log.addEventListener('scroll', () => {
      S.stick = isNearBottom(log);
      if (S.stick) $('jump-new').hidden = true;
    }, { passive: true });
    if (window.ResizeObserver) {
      new ResizeObserver(() => {
        if (S.stick && S.mode === 'game') log.scrollTop = log.scrollHeight;
      }).observe(log);
    }
    window.addEventListener('online', reconnectNow);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') reconnectNow();
    });

    buildLineup();
    showScreen('start');
    renderStart();
    connect();
  }

  // Debug hook (handy for mock harnesses): inject server messages by hand.
  window.WhoIsHuman = { state: S, handle: handle };

  init();
})();
