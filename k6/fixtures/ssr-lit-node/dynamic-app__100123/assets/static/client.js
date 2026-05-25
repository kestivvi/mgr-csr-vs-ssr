//#region node_modules/lit-html/lit-html.js
var e = globalThis, t = (e) => e, n = e.trustedTypes, r = n ? n.createPolicy("lit-html", { createHTML: (e) => e }) : void 0, i = "$lit$", a = `lit$${Math.random().toFixed(9).slice(2)}$`, o = "?" + a, s = `<${o}>`, c = document, l = () => c.createComment(""), u = (e) => e === null || typeof e != "object" && typeof e != "function", d = Array.isArray, f = (e) => d(e) || typeof e?.[Symbol.iterator] == "function", p = "[ 	\n\f\r]", m = /<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g, h = /-->/g, g = />/g, _ = RegExp(`>|${p}(?:([^\\s"'>=/]+)(${p}*=${p}*(?:[^ \t\n\f\r"'\`<>=]|("|')|))|$)`, "g"), v = /'/g, y = /"/g, ee = /^(?:script|style|textarea|title)$/i, b = ((e) => (t, ...n) => ({
	_$litType$: e,
	strings: t,
	values: n
}))(1), x = Symbol.for("lit-noChange"), S = Symbol.for("lit-nothing"), C = /* @__PURE__ */ new WeakMap(), w = c.createTreeWalker(c, 129);
function te(e, t) {
	if (!d(e) || !e.hasOwnProperty("raw")) throw Error("invalid template strings array");
	return r === void 0 ? t : r.createHTML(t);
}
var T = (e, t) => {
	let n = e.length - 1, r = [], o, c = t === 2 ? "<svg>" : t === 3 ? "<math>" : "", l = m;
	for (let t = 0; t < n; t++) {
		let n = e[t], u, d, f = -1, p = 0;
		for (; p < n.length && (l.lastIndex = p, d = l.exec(n), d !== null);) p = l.lastIndex, l === m ? d[1] === "!--" ? l = h : d[1] === void 0 ? d[2] === void 0 ? d[3] !== void 0 && (l = _) : (ee.test(d[2]) && (o = RegExp("</" + d[2], "g")), l = _) : l = g : l === _ ? d[0] === ">" ? (l = o ?? m, f = -1) : d[1] === void 0 ? f = -2 : (f = l.lastIndex - d[2].length, u = d[1], l = d[3] === void 0 ? _ : d[3] === "\"" ? y : v) : l === y || l === v ? l = _ : l === h || l === g ? l = m : (l = _, o = void 0);
		let b = l === _ && e[t + 1].startsWith("/>") ? " " : "";
		c += l === m ? n + s : f >= 0 ? (r.push(u), n.slice(0, f) + i + n.slice(f) + a + b) : n + a + (f === -2 ? t : b);
	}
	return [te(e, c + (e[n] || "<?>") + (t === 2 ? "</svg>" : t === 3 ? "</math>" : "")), r];
}, E = class e {
	constructor({ strings: t, _$litType$: r }, s) {
		let c;
		this.parts = [];
		let u = 0, d = 0, f = t.length - 1, p = this.parts, [m, h] = T(t, r);
		if (this.el = e.createElement(m, s), w.currentNode = this.el.content, r === 2 || r === 3) {
			let e = this.el.content.firstChild;
			e.replaceWith(...e.childNodes);
		}
		for (; (c = w.nextNode()) !== null && p.length < f;) {
			if (c.nodeType === 1) {
				if (c.hasAttributes()) for (let e of c.getAttributeNames()) if (e.endsWith(i)) {
					let t = h[d++], n = c.getAttribute(e).split(a), r = /([.?@])?(.*)/.exec(t);
					p.push({
						type: 1,
						index: u,
						name: r[2],
						strings: n,
						ctor: r[1] === "." ? j : r[1] === "?" ? ne : r[1] === "@" ? re : A
					}), c.removeAttribute(e);
				} else e.startsWith(a) && (p.push({
					type: 6,
					index: u
				}), c.removeAttribute(e));
				if (ee.test(c.tagName)) {
					let e = c.textContent.split(a), t = e.length - 1;
					if (t > 0) {
						c.textContent = n ? n.emptyScript : "";
						for (let n = 0; n < t; n++) c.append(e[n], l()), w.nextNode(), p.push({
							type: 2,
							index: ++u
						});
						c.append(e[t], l());
					}
				}
			} else if (c.nodeType === 8) if (c.data === o) p.push({
				type: 2,
				index: u
			});
			else {
				let e = -1;
				for (; (e = c.data.indexOf(a, e + 1)) !== -1;) p.push({
					type: 7,
					index: u
				}), e += a.length - 1;
			}
			u++;
		}
	}
	static createElement(e, t) {
		let n = c.createElement("template");
		return n.innerHTML = e, n;
	}
};
function D(e, t, n = e, r) {
	if (t === x) return t;
	let i = r === void 0 ? n._$Cl : n._$Co?.[r], a = u(t) ? void 0 : t._$litDirective$;
	return i?.constructor !== a && (i?._$AO?.(!1), a === void 0 ? i = void 0 : (i = new a(e), i._$AT(e, n, r)), r === void 0 ? n._$Cl = i : (n._$Co ??= [])[r] = i), i !== void 0 && (t = D(e, i._$AS(e, t.values), i, r)), t;
}
var O = class {
	constructor(e, t) {
		this._$AV = [], this._$AN = void 0, this._$AD = e, this._$AM = t;
	}
	get parentNode() {
		return this._$AM.parentNode;
	}
	get _$AU() {
		return this._$AM._$AU;
	}
	u(e) {
		let { el: { content: t }, parts: n } = this._$AD, r = (e?.creationScope ?? c).importNode(t, !0);
		w.currentNode = r;
		let i = w.nextNode(), a = 0, o = 0, s = n[0];
		for (; s !== void 0;) {
			if (a === s.index) {
				let t;
				s.type === 2 ? t = new k(i, i.nextSibling, this, e) : s.type === 1 ? t = new s.ctor(i, s.name, s.strings, this, e) : s.type === 6 && (t = new ie(i, this, e)), this._$AV.push(t), s = n[++o];
			}
			a !== s?.index && (i = w.nextNode(), a++);
		}
		return w.currentNode = c, r;
	}
	p(e) {
		let t = 0;
		for (let n of this._$AV) n !== void 0 && (n.strings === void 0 ? n._$AI(e[t]) : (n._$AI(e, n, t), t += n.strings.length - 2)), t++;
	}
}, k = class e {
	get _$AU() {
		return this._$AM?._$AU ?? this._$Cv;
	}
	constructor(e, t, n, r) {
		this.type = 2, this._$AH = S, this._$AN = void 0, this._$AA = e, this._$AB = t, this._$AM = n, this.options = r, this._$Cv = r?.isConnected ?? !0;
	}
	get parentNode() {
		let e = this._$AA.parentNode, t = this._$AM;
		return t !== void 0 && e?.nodeType === 11 && (e = t.parentNode), e;
	}
	get startNode() {
		return this._$AA;
	}
	get endNode() {
		return this._$AB;
	}
	_$AI(e, t = this) {
		e = D(this, e, t), u(e) ? e === S || e == null || e === "" ? (this._$AH !== S && this._$AR(), this._$AH = S) : e !== this._$AH && e !== x && this._(e) : e._$litType$ === void 0 ? e.nodeType === void 0 ? f(e) ? this.k(e) : this._(e) : this.T(e) : this.$(e);
	}
	O(e) {
		return this._$AA.parentNode.insertBefore(e, this._$AB);
	}
	T(e) {
		this._$AH !== e && (this._$AR(), this._$AH = this.O(e));
	}
	_(e) {
		this._$AH !== S && u(this._$AH) ? this._$AA.nextSibling.data = e : this.T(c.createTextNode(e)), this._$AH = e;
	}
	$(e) {
		let { values: t, _$litType$: n } = e, r = typeof n == "number" ? this._$AC(e) : (n.el === void 0 && (n.el = E.createElement(te(n.h, n.h[0]), this.options)), n);
		if (this._$AH?._$AD === r) this._$AH.p(t);
		else {
			let e = new O(r, this), n = e.u(this.options);
			e.p(t), this.T(n), this._$AH = e;
		}
	}
	_$AC(e) {
		let t = C.get(e.strings);
		return t === void 0 && C.set(e.strings, t = new E(e)), t;
	}
	k(t) {
		d(this._$AH) || (this._$AH = [], this._$AR());
		let n = this._$AH, r, i = 0;
		for (let a of t) i === n.length ? n.push(r = new e(this.O(l()), this.O(l()), this, this.options)) : r = n[i], r._$AI(a), i++;
		i < n.length && (this._$AR(r && r._$AB.nextSibling, i), n.length = i);
	}
	_$AR(e = this._$AA.nextSibling, n) {
		for (this._$AP?.(!1, !0, n); e !== this._$AB;) {
			let n = t(e).nextSibling;
			t(e).remove(), e = n;
		}
	}
	setConnected(e) {
		this._$AM === void 0 && (this._$Cv = e, this._$AP?.(e));
	}
}, A = class {
	get tagName() {
		return this.element.tagName;
	}
	get _$AU() {
		return this._$AM._$AU;
	}
	constructor(e, t, n, r, i) {
		this.type = 1, this._$AH = S, this._$AN = void 0, this.element = e, this.name = t, this._$AM = r, this.options = i, n.length > 2 || n[0] !== "" || n[1] !== "" ? (this._$AH = Array(n.length - 1).fill(/* @__PURE__ */ new String()), this.strings = n) : this._$AH = S;
	}
	_$AI(e, t = this, n, r) {
		let i = this.strings, a = !1;
		if (i === void 0) e = D(this, e, t, 0), a = !u(e) || e !== this._$AH && e !== x, a && (this._$AH = e);
		else {
			let r = e, o, s;
			for (e = i[0], o = 0; o < i.length - 1; o++) s = D(this, r[n + o], t, o), s === x && (s = this._$AH[o]), a ||= !u(s) || s !== this._$AH[o], s === S ? e = S : e !== S && (e += (s ?? "") + i[o + 1]), this._$AH[o] = s;
		}
		a && !r && this.j(e);
	}
	j(e) {
		e === S ? this.element.removeAttribute(this.name) : this.element.setAttribute(this.name, e ?? "");
	}
}, j = class extends A {
	constructor() {
		super(...arguments), this.type = 3;
	}
	j(e) {
		this.element[this.name] = e === S ? void 0 : e;
	}
}, ne = class extends A {
	constructor() {
		super(...arguments), this.type = 4;
	}
	j(e) {
		this.element.toggleAttribute(this.name, !!e && e !== S);
	}
}, re = class extends A {
	constructor(e, t, n, r, i) {
		super(e, t, n, r, i), this.type = 5;
	}
	_$AI(e, t = this) {
		if ((e = D(this, e, t, 0) ?? S) === x) return;
		let n = this._$AH, r = e === S && n !== S || e.capture !== n.capture || e.once !== n.once || e.passive !== n.passive, i = e !== S && (n === S || r);
		r && this.element.removeEventListener(this.name, this, n), i && this.element.addEventListener(this.name, this, e), this._$AH = e;
	}
	handleEvent(e) {
		typeof this._$AH == "function" ? this._$AH.call(this.options?.host ?? this.element, e) : this._$AH.handleEvent(e);
	}
}, ie = class {
	constructor(e, t, n) {
		this.element = e, this.type = 6, this._$AN = void 0, this._$AM = t, this.options = n;
	}
	get _$AU() {
		return this._$AM._$AU;
	}
	_$AI(e) {
		D(this, e);
	}
}, M = {
	M: i,
	P: a,
	A: o,
	C: 1,
	L: T,
	R: O,
	D: f,
	V: D,
	I: k,
	H: A,
	N: ne,
	U: re,
	B: j,
	F: ie
}, ae = e.litHtmlPolyfillSupport;
ae?.(E, k), (e.litHtmlVersions ??= []).push("3.3.2");
var oe = (e, t, n) => {
	let r = n?.renderBefore ?? t, i = r._$litPart$;
	if (i === void 0) {
		let e = n?.renderBefore ?? null;
		r._$litPart$ = i = new k(t.insertBefore(l(), e), e, void 0, n ?? {});
	}
	return i._$AI(e), i;
}, N = null, se = {
	boundAttributeSuffix: M.M,
	marker: M.P,
	markerMatch: M.A,
	HTML_RESULT: M.C,
	getTemplateHtml: M.L,
	overrideDirectiveResolve: (e, t) => class extends e {
		_$AS(e, n) {
			return t(this, n);
		}
	},
	patchDirectiveResolve: (e, t) => {
		if (e.prototype._$AS.name !== t.name) {
			N ??= e.prototype._$AS.name;
			for (let n = e.prototype; n !== Object.prototype; n = Object.getPrototypeOf(n)) if (n.hasOwnProperty(N)) return void (n[N] = t);
			throw Error("Internal error: It is possible that both dev mode and production mode Lit was mixed together during SSR. Please comment on the issue: https://github.com/lit/lit/issues/4527");
		}
	},
	setDirectiveClass(e, t) {
		e._$litDirective$ = t;
	},
	getAttributePartCommittedValue: (e, t, n) => {
		let r = x;
		return e.j = (e) => r = e, e._$AI(t, e, n), r;
	},
	connectedDisconnectable: (e) => ({
		...e,
		_$AU: !0
	}),
	resolveDirective: M.V,
	AttributePart: M.H,
	PropertyPart: M.B,
	BooleanAttributePart: M.N,
	EventPart: M.U,
	ElementPart: M.F,
	TemplateInstance: M.R,
	isIterable: M.D,
	ChildPart: M.I
}, P = {
	ATTRIBUTE: 1,
	CHILD: 2,
	PROPERTY: 3,
	BOOLEAN_ATTRIBUTE: 4,
	EVENT: 5,
	ELEMENT: 6
}, { I: ce } = M, le = (e) => e === null || typeof e != "object" && typeof e != "function", ue = (e, t) => t === void 0 ? e?._$litType$ !== void 0 : e?._$litType$ === t, de = (e) => e?._$litType$?.h != null, fe = (e) => e.strings === void 0, { TemplateInstance: pe, isIterable: me, resolveDirective: F, ChildPart: I, ElementPart: he } = se, ge = (e, t, n = {}) => {
	if (t._$litPart$ !== void 0) throw Error("container already contains a live render");
	let r, i, a, o = [], s = document.createTreeWalker(t, NodeFilter.SHOW_COMMENT), c;
	for (; (c = s.nextNode()) !== null;) {
		let t = c.data;
		if (t.startsWith("lit-part")) {
			if (o.length === 0 && r !== void 0) throw Error(`There must be only one root part per container. Found a part marker (${c}) when we already have a root part marker (${i})`);
			a = _e(e, c, o, n), r === void 0 && (r = a), i ??= c;
		} else if (t.startsWith("lit-node")) ye(c, o, n);
		else if (t.startsWith("/lit-part")) {
			if (o.length === 1 && a !== r) throw Error("internal error");
			a = ve(c, a, o);
		}
	}
	if (r === void 0) {
		let e = t instanceof ShadowRoot ? "{container.host.localName}'s shadow root" : t instanceof DocumentFragment ? "DocumentFragment" : t.localName;
		console.error(`There should be exactly one root part in a render container, but we didn't find any in ${e}.`);
	}
	t._$litPart$ = r;
}, _e = (e, t, n, r) => {
	let i, a;
	if (n.length === 0) a = new I(t, null, void 0, r), i = e;
	else {
		let e = n[n.length - 1];
		if (e.type === "template-instance") a = new I(t, null, e.instance, r), e.instance._$AV.push(a), i = e.result.values[e.instancePartIndex++], e.templatePartIndex++;
		else if (e.type === "iterable") {
			a = new I(t, null, e.part, r);
			let n = e.iterator.next();
			if (n.done) throw i = void 0, e.done = !0, Error("Unhandled shorter than expected iterable");
			i = n.value, e.part._$AH.push(a);
		} else a = new I(t, null, e.part, r);
	}
	if (i = F(a, i), i === x) n.push({
		part: a,
		type: "leaf"
	});
	else if (le(i)) n.push({
		part: a,
		type: "leaf"
	}), a._$AH = i;
	else if (ue(i)) {
		if (de(i)) throw Error("compiled templates are not supported");
		let e = "lit-part " + be(i);
		if (t.data !== e) throw Error("Hydration value mismatch: Unexpected TemplateResult rendered to part");
		{
			let e = new pe(I.prototype._$AC(i), a);
			n.push({
				type: "template-instance",
				instance: e,
				part: a,
				templatePartIndex: 0,
				instancePartIndex: 0,
				result: i
			}), a._$AH = e;
		}
	} else me(i) ? (n.push({
		part: a,
		type: "iterable",
		value: i,
		iterator: i[Symbol.iterator](),
		done: !1
	}), a._$AH = []) : (n.push({
		part: a,
		type: "leaf"
	}), a._$AH = i ?? "");
	return a;
}, ve = (e, t, n) => {
	if (t === void 0) throw Error("unbalanced part marker");
	t._$AB = e;
	let r = n.pop();
	if (r.type === "iterable" && !r.iterator.next().done) throw Error("unexpected longer than expected iterable");
	if (n.length > 0) return n[n.length - 1].part;
}, ye = (e, t, n) => {
	let r = /lit-node (\d+)/.exec(e.data), i = parseInt(r[1]), a = e.nextElementSibling;
	if (a === null) throw Error("could not find node for attribute parts");
	a.removeAttribute("defer-hydration");
	let o = t[t.length - 1];
	if (o.type !== "template-instance") throw Error("Hydration value mismatch: Primitive found where TemplateResult expected. This usually occurs due to conditional rendering that resulted in a different value or template being rendered between the server and client.");
	{
		let e = o.instance;
		for (;;) {
			let t = e._$AD.parts[o.templatePartIndex];
			if (t === void 0 || t.type !== P.ATTRIBUTE && t.type !== P.ELEMENT || t.index !== i) break;
			if (t.type === P.ATTRIBUTE) {
				let r = new t.ctor(a, t.name, t.strings, o.instance, n), i = fe(r) ? o.result.values[o.instancePartIndex] : o.result.values, s = !(r.type === P.EVENT || r.type === P.PROPERTY);
				r._$AI(i, r, o.instancePartIndex, s), o.instancePartIndex += t.strings.length - 1, e._$AV.push(r);
			} else {
				let t = new he(a, o.instance, n);
				F(t, o.result.values[o.instancePartIndex++]), e._$AV.push(t);
			}
			o.templatePartIndex++;
		}
	}
}, L = /* @__PURE__ */ new WeakMap(), be = (e) => {
	let t = L.get(e.strings);
	if (t !== void 0) return t;
	let n = new Uint32Array(2).fill(5381);
	for (let t of e.strings) for (let e = 0; e < t.length; e++) n[e % 2] = 33 * n[e % 2] ^ t.charCodeAt(e);
	let r = String.fromCharCode(...new Uint8Array(n.buffer));
	return t = btoa(r), L.set(e.strings, t), t;
};
//#endregion
//#region node_modules/@lit-labs/ssr-client/lit-element-hydrate-support.js
globalThis.litElementHydrateSupport = ({ LitElement: e }) => {
	let t = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(e), "observedAttributes").get;
	Object.defineProperty(e, "observedAttributes", { get() {
		return [...t.call(this), "defer-hydration"];
	} });
	let n = e.prototype.attributeChangedCallback;
	e.prototype.attributeChangedCallback = function(e, t, i) {
		e === "defer-hydration" && i === null && r.call(this), n.call(this, e, t, i);
	};
	let r = e.prototype.connectedCallback;
	e.prototype.connectedCallback = function() {
		this.hasAttribute("defer-hydration") || r.call(this);
	};
	let i = e.prototype.createRenderRoot;
	e.prototype.createRenderRoot = function() {
		return this.shadowRoot ? (this._$AG = !0, this.shadowRoot) : i.call(this);
	};
	let a = Object.getPrototypeOf(e.prototype).update;
	e.prototype.update = function(e) {
		let t = this.render();
		if (a.call(this, e), this._$AG) {
			this._$AG = !1;
			for (let e of this.getAttributeNames()) if (e.startsWith("hydrate-internals-")) {
				let t = e.slice(18);
				this.removeAttribute(t), this.removeAttribute(e);
			}
			ge(t, this.renderRoot, this.renderOptions);
		} else oe(t, this.renderRoot, this.renderOptions);
	};
};
//#endregion
//#region node_modules/@lit/reactive-element/css-tag.js
var R = globalThis, z = R.ShadowRoot && (R.ShadyCSS === void 0 || R.ShadyCSS.nativeShadow) && "adoptedStyleSheets" in Document.prototype && "replace" in CSSStyleSheet.prototype, B = Symbol(), V = /* @__PURE__ */ new WeakMap(), H = class {
	constructor(e, t, n) {
		if (this._$cssResult$ = !0, n !== B) throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");
		this.cssText = e, this.t = t;
	}
	get styleSheet() {
		let e = this.o, t = this.t;
		if (z && e === void 0) {
			let n = t !== void 0 && t.length === 1;
			n && (e = V.get(t)), e === void 0 && ((this.o = e = new CSSStyleSheet()).replaceSync(this.cssText), n && V.set(t, e));
		}
		return e;
	}
	toString() {
		return this.cssText;
	}
}, xe = (e) => new H(typeof e == "string" ? e : e + "", void 0, B), Se = (e, ...t) => new H(e.length === 1 ? e[0] : t.reduce((t, n, r) => t + ((e) => {
	if (!0 === e._$cssResult$) return e.cssText;
	if (typeof e == "number") return e;
	throw Error("Value passed to 'css' function must be a 'css' function result: " + e + ". Use 'unsafeCSS' to pass non-literal values, but take care to ensure page security.");
})(n) + e[r + 1], e[0]), e, B), Ce = (e, t) => {
	if (z) e.adoptedStyleSheets = t.map((e) => e instanceof CSSStyleSheet ? e : e.styleSheet);
	else for (let n of t) {
		let t = document.createElement("style"), r = R.litNonce;
		r !== void 0 && t.setAttribute("nonce", r), t.textContent = n.cssText, e.appendChild(t);
	}
}, U = z ? (e) => e : (e) => e instanceof CSSStyleSheet ? ((e) => {
	let t = "";
	for (let n of e.cssRules) t += n.cssText;
	return xe(t);
})(e) : e, { is: we, defineProperty: Te, getOwnPropertyDescriptor: Ee, getOwnPropertyNames: De, getOwnPropertySymbols: Oe, getPrototypeOf: ke } = Object, W = globalThis, G = W.trustedTypes, Ae = G ? G.emptyScript : "", je = W.reactiveElementPolyfillSupport, K = (e, t) => e, q = {
	toAttribute(e, t) {
		switch (t) {
			case Boolean:
				e = e ? Ae : null;
				break;
			case Object:
			case Array: e = e == null ? e : JSON.stringify(e);
		}
		return e;
	},
	fromAttribute(e, t) {
		let n = e;
		switch (t) {
			case Boolean:
				n = e !== null;
				break;
			case Number:
				n = e === null ? null : Number(e);
				break;
			case Object:
			case Array: try {
				n = JSON.parse(e);
			} catch {
				n = null;
			}
		}
		return n;
	}
}, J = (e, t) => !we(e, t), Y = {
	attribute: !0,
	type: String,
	converter: q,
	reflect: !1,
	useDefault: !1,
	hasChanged: J
};
Symbol.metadata ??= Symbol("metadata"), W.litPropertyMetadata ??= /* @__PURE__ */ new WeakMap();
var X = class extends HTMLElement {
	static addInitializer(e) {
		this._$Ei(), (this.l ??= []).push(e);
	}
	static get observedAttributes() {
		return this.finalize(), this._$Eh && [...this._$Eh.keys()];
	}
	static createProperty(e, t = Y) {
		if (t.state && (t.attribute = !1), this._$Ei(), this.prototype.hasOwnProperty(e) && ((t = Object.create(t)).wrapped = !0), this.elementProperties.set(e, t), !t.noAccessor) {
			let n = Symbol(), r = this.getPropertyDescriptor(e, n, t);
			r !== void 0 && Te(this.prototype, e, r);
		}
	}
	static getPropertyDescriptor(e, t, n) {
		let { get: r, set: i } = Ee(this.prototype, e) ?? {
			get() {
				return this[t];
			},
			set(e) {
				this[t] = e;
			}
		};
		return {
			get: r,
			set(t) {
				let a = r?.call(this);
				i?.call(this, t), this.requestUpdate(e, a, n);
			},
			configurable: !0,
			enumerable: !0
		};
	}
	static getPropertyOptions(e) {
		return this.elementProperties.get(e) ?? Y;
	}
	static _$Ei() {
		if (this.hasOwnProperty(K("elementProperties"))) return;
		let e = ke(this);
		e.finalize(), e.l !== void 0 && (this.l = [...e.l]), this.elementProperties = new Map(e.elementProperties);
	}
	static finalize() {
		if (this.hasOwnProperty(K("finalized"))) return;
		if (this.finalized = !0, this._$Ei(), this.hasOwnProperty(K("properties"))) {
			let e = this.properties, t = [...De(e), ...Oe(e)];
			for (let n of t) this.createProperty(n, e[n]);
		}
		let e = this[Symbol.metadata];
		if (e !== null) {
			let t = litPropertyMetadata.get(e);
			if (t !== void 0) for (let [e, n] of t) this.elementProperties.set(e, n);
		}
		this._$Eh = /* @__PURE__ */ new Map();
		for (let [e, t] of this.elementProperties) {
			let n = this._$Eu(e, t);
			n !== void 0 && this._$Eh.set(n, e);
		}
		this.elementStyles = this.finalizeStyles(this.styles);
	}
	static finalizeStyles(e) {
		let t = [];
		if (Array.isArray(e)) {
			let n = new Set(e.flat(Infinity).reverse());
			for (let e of n) t.unshift(U(e));
		} else e !== void 0 && t.push(U(e));
		return t;
	}
	static _$Eu(e, t) {
		let n = t.attribute;
		return !1 === n ? void 0 : typeof n == "string" ? n : typeof e == "string" ? e.toLowerCase() : void 0;
	}
	constructor() {
		super(), this._$Ep = void 0, this.isUpdatePending = !1, this.hasUpdated = !1, this._$Em = null, this._$Ev();
	}
	_$Ev() {
		this._$ES = new Promise((e) => this.enableUpdating = e), this._$AL = /* @__PURE__ */ new Map(), this._$E_(), this.requestUpdate(), this.constructor.l?.forEach((e) => e(this));
	}
	addController(e) {
		(this._$EO ??= /* @__PURE__ */ new Set()).add(e), this.renderRoot !== void 0 && this.isConnected && e.hostConnected?.();
	}
	removeController(e) {
		this._$EO?.delete(e);
	}
	_$E_() {
		let e = /* @__PURE__ */ new Map(), t = this.constructor.elementProperties;
		for (let n of t.keys()) this.hasOwnProperty(n) && (e.set(n, this[n]), delete this[n]);
		e.size > 0 && (this._$Ep = e);
	}
	createRenderRoot() {
		let e = this.shadowRoot ?? this.attachShadow(this.constructor.shadowRootOptions);
		return Ce(e, this.constructor.elementStyles), e;
	}
	connectedCallback() {
		this.renderRoot ??= this.createRenderRoot(), this.enableUpdating(!0), this._$EO?.forEach((e) => e.hostConnected?.());
	}
	enableUpdating(e) {}
	disconnectedCallback() {
		this._$EO?.forEach((e) => e.hostDisconnected?.());
	}
	attributeChangedCallback(e, t, n) {
		this._$AK(e, n);
	}
	_$ET(e, t) {
		let n = this.constructor.elementProperties.get(e), r = this.constructor._$Eu(e, n);
		if (r !== void 0 && !0 === n.reflect) {
			let i = (n.converter?.toAttribute === void 0 ? q : n.converter).toAttribute(t, n.type);
			this._$Em = e, i == null ? this.removeAttribute(r) : this.setAttribute(r, i), this._$Em = null;
		}
	}
	_$AK(e, t) {
		let n = this.constructor, r = n._$Eh.get(e);
		if (r !== void 0 && this._$Em !== r) {
			let e = n.getPropertyOptions(r), i = typeof e.converter == "function" ? { fromAttribute: e.converter } : e.converter?.fromAttribute === void 0 ? q : e.converter;
			this._$Em = r;
			let a = i.fromAttribute(t, e.type);
			this[r] = a ?? this._$Ej?.get(r) ?? a, this._$Em = null;
		}
	}
	requestUpdate(e, t, n, r = !1, i) {
		if (e !== void 0) {
			let a = this.constructor;
			if (!1 === r && (i = this[e]), n ??= a.getPropertyOptions(e), !((n.hasChanged ?? J)(i, t) || n.useDefault && n.reflect && i === this._$Ej?.get(e) && !this.hasAttribute(a._$Eu(e, n)))) return;
			this.C(e, t, n);
		}
		!1 === this.isUpdatePending && (this._$ES = this._$EP());
	}
	C(e, t, { useDefault: n, reflect: r, wrapped: i }, a) {
		n && !(this._$Ej ??= /* @__PURE__ */ new Map()).has(e) && (this._$Ej.set(e, a ?? t ?? this[e]), !0 !== i || a !== void 0) || (this._$AL.has(e) || (this.hasUpdated || n || (t = void 0), this._$AL.set(e, t)), !0 === r && this._$Em !== e && (this._$Eq ??= /* @__PURE__ */ new Set()).add(e));
	}
	async _$EP() {
		this.isUpdatePending = !0;
		try {
			await this._$ES;
		} catch (e) {
			Promise.reject(e);
		}
		let e = this.scheduleUpdate();
		return e != null && await e, !this.isUpdatePending;
	}
	scheduleUpdate() {
		return this.performUpdate();
	}
	performUpdate() {
		if (!this.isUpdatePending) return;
		if (!this.hasUpdated) {
			if (this.renderRoot ??= this.createRenderRoot(), this._$Ep) {
				for (let [e, t] of this._$Ep) this[e] = t;
				this._$Ep = void 0;
			}
			let e = this.constructor.elementProperties;
			if (e.size > 0) for (let [t, n] of e) {
				let { wrapped: e } = n, r = this[t];
				!0 !== e || this._$AL.has(t) || r === void 0 || this.C(t, void 0, n, r);
			}
		}
		let e = !1, t = this._$AL;
		try {
			e = this.shouldUpdate(t), e ? (this.willUpdate(t), this._$EO?.forEach((e) => e.hostUpdate?.()), this.update(t)) : this._$EM();
		} catch (t) {
			throw e = !1, this._$EM(), t;
		}
		e && this._$AE(t);
	}
	willUpdate(e) {}
	_$AE(e) {
		this._$EO?.forEach((e) => e.hostUpdated?.()), this.hasUpdated || (this.hasUpdated = !0, this.firstUpdated(e)), this.updated(e);
	}
	_$EM() {
		this._$AL = /* @__PURE__ */ new Map(), this.isUpdatePending = !1;
	}
	get updateComplete() {
		return this.getUpdateComplete();
	}
	getUpdateComplete() {
		return this._$ES;
	}
	shouldUpdate(e) {
		return !0;
	}
	update(e) {
		this._$Eq &&= this._$Eq.forEach((e) => this._$ET(e, this[e])), this._$EM();
	}
	updated(e) {}
	firstUpdated(e) {}
};
X.elementStyles = [], X.shadowRootOptions = { mode: "open" }, X[K("elementProperties")] = /* @__PURE__ */ new Map(), X[K("finalized")] = /* @__PURE__ */ new Map(), je?.({ ReactiveElement: X }), (W.reactiveElementVersions ??= []).push("2.1.2");
//#endregion
//#region node_modules/lit-element/lit-element.js
var Z = globalThis, Q = class extends X {
	constructor() {
		super(...arguments), this.renderOptions = { host: this }, this._$Do = void 0;
	}
	createRenderRoot() {
		let e = super.createRenderRoot();
		return this.renderOptions.renderBefore ??= e.firstChild, e;
	}
	update(e) {
		let t = this.render();
		this.hasUpdated || (this.renderOptions.isConnected = this.isConnected), super.update(e), this._$Do = oe(t, this.renderRoot, this.renderOptions);
	}
	connectedCallback() {
		super.connectedCallback(), this._$Do?.setConnected(!0);
	}
	disconnectedCallback() {
		super.disconnectedCallback(), this._$Do?.setConnected(!1);
	}
	render() {
		return x;
	}
};
Q._$litElement$ = !0, Q.finalized = !0, Z.litElementHydrateSupport?.({ LitElement: Q });
var Me = Z.litElementPolyfillSupport;
Me?.({ LitElement: Q }), (Z.litElementVersions ??= []).push("4.2.2");
//#endregion
//#region node_modules/@lit/reactive-element/decorators/custom-element.js
var Ne = (e) => (t, n) => {
	n === void 0 ? customElements.define(e, t) : n.addInitializer(() => {
		customElements.define(e, t);
	});
}, Pe = {
	attribute: !0,
	type: String,
	converter: q,
	reflect: !1,
	hasChanged: J
}, Fe = (e = Pe, t, n) => {
	let { kind: r, metadata: i } = n, a = globalThis.litPropertyMetadata.get(i);
	if (a === void 0 && globalThis.litPropertyMetadata.set(i, a = /* @__PURE__ */ new Map()), r === "setter" && ((e = Object.create(e)).wrapped = !0), a.set(n.name, e), r === "accessor") {
		let { name: r } = n;
		return {
			set(n) {
				let i = t.get.call(this);
				t.set.call(this, n), this.requestUpdate(r, i, e, !0, n);
			},
			init(t) {
				return t !== void 0 && this.C(r, void 0, e, t), t;
			}
		};
	}
	if (r === "setter") {
		let { name: r } = n;
		return function(n) {
			let i = this[r];
			t.call(this, n), this.requestUpdate(r, i, e, !0, n);
		};
	}
	throw Error("Unsupported decorator location: " + r);
};
function Ie(e) {
	return (t, n) => typeof n == "object" ? Fe(e, t, n) : ((e, t, n) => {
		let r = t.hasOwnProperty(n);
		return t.constructor.createProperty(n, e), r ? Object.getOwnPropertyDescriptor(t, n) : void 0;
	})(e, t, n);
}
//#endregion
//#region src/styles.ts
var Le = Se`
  h1 {
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 1rem;
      color: #222;
  }

  p {
      font-size: 1.2rem;
      color: #666;
      margin-bottom: 1.5rem;
  }

  button {
      background: #007bff;
      color: white;
      border: none;
      padding: 0.8rem 1.5rem;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
  }

  button:hover {
      background: #0056b3;
  }

  button:active {
      transform: scale(0.98);
  }
`;
//#endregion
//#region \0@oxc-project+runtime@0.127.0/helpers/decorate.js
function Re(e, t, n, r) {
	var i = arguments.length, a = i < 3 ? t : r === null ? r = Object.getOwnPropertyDescriptor(t, n) : r, o;
	if (typeof Reflect == "object" && typeof Reflect.decorate == "function") a = Reflect.decorate(e, t, n, r);
	else for (var s = e.length - 1; s >= 0; s--) (o = e[s]) && (a = (i < 3 ? o(a) : i > 3 ? o(t, n, a) : o(t, n)) || a);
	return i > 3 && a && Object.defineProperty(t, n, a), a;
}
//#endregion
//#region src/components/counter-element.ts
var $ = class extends Q {
	static styles = Le;
	count = 0;
	render() {
		return b`
      <div>
        <p>Count: ${this.count}</p>
        <button @click=${() => this.count++}>Increment</button>
      </div>
    `;
	}
};
//#endregion
//#region src/client.ts
Re([Ie({ type: Number })], $.prototype, "count", void 0), $ = Re([Ne("counter-element")], $), console.log("Hydration complete");
//#endregion
