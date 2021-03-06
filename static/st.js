!(function () {
    "use strict";
    const A = console;
    var selections = [];
    var suggestions = [];
    var canceled = false;
    var start_time = new Date().getTime()

    // TODO: Configurable defaults
    const TAB_ENABLED = true;
    const DEBOUNCE_DELAY_MILLIS = 1200;
    const MIN_TEXT_LENGTH = 100;
    const REPORT_SELECTIONS_ON_SUBMIT = true;
    //const ENDPOINT = "http://127.0.0.1:5000";
    const ENDPOINT = "http://so-srv1.iem.technion.ac.il:9000";
    const COMPLETION_MSG = "Thank you!\nText was submitted. Please return to system to complete the survey";
    const COMPLETION_LINK = "https://technioniit.eu.qualtrics.com/jfe/form/SV_8eqpOdQvlrv7nHE";

    function debounce(func, wait, immediate) {
        var timeout;
        return function() {
            var context = this, args = arguments;
            var later = function() {
                timeout = null;
                if (!immediate) func.apply(context, args);
            };
            var callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func.apply(context, args);
        };
    };

    let triggerPredictions = debounce(function(f) {
        f();
    }, DEBOUNCE_DELAY_MILLIS);

    class e {
        path(A) {
            return `${e.ENDPOINT}/${A}`;
        }

        async postAutocomplete(A) {
            const e = this.path(`autocomplete`),
                t = await fetch(e, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(A) });
            return await t.json();

        }

        async postSubmission(text) {
            canceled = true;
            const p = this.path(`submit`),
                t = await fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(
                    {
                        userid: window.appUserId,
                        completionEnabled: window.completionEnabled,
                        text: text,
                        // timestamp: new Date().getTime(),
                        start_time: start_time,
                        end_time: new Date().getTime(),
//                        selections: selections,
                        suggestions: suggestions
                    })});

//            const data = await t.json();
//            if (data.status == "error"){
//                alert("Save Error");
//            }else{
//                selections = [];
//                suggestions = [];
//                a.setText('');
//                alert(COMPLETION_MSG);
//            }
//            return 'ok'
            return await t.json();

        }

        async postWithSettings(A) {
            const e = document.querySelector(".decoder-settings .setting.model_size .js-val").textContent || void 0,
                t = (A) => {
                    const e = document.querySelector(A);
                    if (e && e.textContent) return Number(e.textContent);
                },
                r = t(".decoder-settings .setting.top_p .js-val"),
                n = t(".decoder-settings .setting.temperature .js-val"),
                s = t(".decoder-settings .setting.step_size .js-val"),
                i = t(".decoder-settings .setting.kl_scale .js-val"),
                o = t(".decoder-settings .setting.gm_scale .js-val"),
                B = t(".decoder-settings .setting.num_iterations .js-val"),
                a = t(".decoder-settings .setting.gen_length .js-val"),
                c = t(".decoder-settings .setting.max_time .js-val"),
                l = (document.querySelector(".decoder-settings input[name=bow_or_discrim]:checked") || {}).value,
                Q = (document.querySelector(".decoder-settings input[name=use_sampling]") || {}).checked;
//            return this.postAutocomplete({ ...A, model_size: e, top_p: r, temperature: n, step_size: s, kl_scale: i, gm_scale: o, num_iterations: B, gen_length: a, max_time: c, bow_or_discrim: l, use_sampling: Q });
                return this.postAutocomplete({ ...A, userid: window.appUserId , timestamp: new Date().getTime() ,model_size: e, top_p: r, temperature: n, step_size: s, kl_scale: i, gm_scale: o, num_iterations: B, gen_length: a, max_time: c, bow_or_discrim: l, use_sampling: Q });
        }
        async postEdit(A) {
            const e = window.doc;
            if (!e || !e.longId) throw new Error("invalid doc");
            const t = `/edit/${e.model}/${e.longId}/${e.shortId}`;
            return (await fetch(t, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(A) })).ok;
        }
        async postDuplicate() {
            const e = window.doc;
            if (!e || !e.shortId) throw new Error("invalid doc");
            const t = `/duplicate/${e.shortId}`,
                r = await fetch(t, { method: "POST", headers: { "Content-Type": "application/json" } }),
                n = await r.text();
            return A.log("[new url]", n), n;
        }
    }
    //(e.ENDPOINT = "https://transformer.huggingface.co"), (e.shared = new e());
    //(e.ENDPOINT = "http://f7e0189baf6d.ngrok.io/"), (e.shared = new e());
    (e.ENDPOINT = ENDPOINT), (e.shared = new e());
    class t {
        constructor(A) {
            (this.quill = A),
                (this.isOpen = !1),
                (this.itemIndex = 0),
                (this.mentionCharPos = void 0),
                (this.cursorPos = void 0),
                (this.values = []),
                (this.suspendMouseEnter = !1),
                (this.options = {
                    source: (A, e, t) => {},
                    renderItem: (A, e) => `${A.value}`,
                    onSelect: (A, e) => {
                        e(A);
                    },
                    mentionDenotationChars: ["@"],
                    showDenotationChar: !0,
                    allowedChars: /^[a-zA-Z0-9_]*$/,
                    minChars: 0,
                    maxChars: 31,
                    offsetTop: 2,
                    offsetLeft: 0,
                    isolateCharacter: !1,
                    fixMentionsToQuill: !1,
                    defaultMenuOrientation: "bottom",
                    dataAttributes: ["id", "value", "denotationChar", "link", "target"],
                    linkTarget: "_blank",
                    onOpen: () => !0,
                    onClose: () => !0,
                    listItemClass: "ql-mention-list-item",
                    mentionContainerClass: "ql-mention-list-container",
                    mentionListClass: "ql-mention-list",
                }),
                (this.mentionContainer = document.createElement("div")),
                (this.mentionList = document.createElement("ul")),
                (this.mentionContainer.className = this.options.mentionContainerClass),
                (this.mentionContainer.style.cssText = "display: none; position: absolute;"),
                (this.mentionContainer.onmousemove = this.onContainerMouseMove.bind(this)),
                this.options.fixMentionsToQuill && (this.mentionContainer.style.width = "auto"),
                (this.mentionList.className = this.options.mentionListClass),
                this.mentionContainer.appendChild(this.mentionList),
                this.quill.container.appendChild(this.mentionContainer),
                A.on("text-change", this.onTextChange.bind(this)),
                A.on("selection-change", this.onSelectionChange.bind(this)),
                A.keyboard.addBinding({ key: t.Keys.ENTER }, this.selectHandler.bind(this)),
                A.keyboard.bindings[t.Keys.ENTER].unshift(A.keyboard.bindings[t.Keys.ENTER].pop()),
                A.keyboard.addBinding({ key: t.Keys.ESCAPE }, this.escapeHandler.bind(this)),
                A.keyboard.addBinding({ key: t.Keys.UP }, this.upHandler.bind(this)),
                A.keyboard.addBinding({ key: t.Keys.DOWN }, this.downHandler.bind(this)),
                document.addEventListener("keypress", (A) => {
                    this.quill.hasFocus() &&
                        setTimeout(() => {
                            this.setCursorPos(), this.quill.removeFormat(this.cursorPos - 1, 1, "silent");
                        }, 0);
                });
                this.quill.focus();
                setTimeout(() => {
                    this.quill.setText('');
                }, 0);
        }
        selectHandler() {
            this.postAutocompleteSelection(this.getItemData()).then(result => {
                // console.log('selection submission result:', result);
            });
            return !this.isOpen || (this.selectItem(), !1);
        }
        escapeHandler() {
            return !this.isOpen || (this.hideMentionList(), !1);
        }
        upHandler() {
            return !this.isOpen || (this.prevItem(), !1);
        }
        downHandler() {
            return !this.isOpen || (this.nextItem(), !1);
        }
        showMentionList() {
            (this.mentionContainer.style.visibility = "hidden"), (this.mentionContainer.style.display = ""), this.setMentionContainerPosition(), this.setIsOpen(!0);
        }
        hideMentionList() {
            (this.mentionContainer.style.display = "none"), this.setIsOpen(!1);
        }
        highlightItem(A = !0) {
            const e = Array.from(this.mentionList.childNodes);
            for (const A of e) A.classList.remove("selected");
            if ((e[this.itemIndex].classList.add("selected"), A)) {
                const A = e[this.itemIndex].offsetHeight,
                    t = this.itemIndex * A,
                    r = this.mentionContainer.scrollTop,
                    n = r + this.mentionContainer.offsetHeight;
                t < r ? (this.mentionContainer.scrollTop = t) : t > n - A && (this.mentionContainer.scrollTop += t - n + A);
            }
        }
        getItemData() {
            const A = this.mentionList.childNodes[this.itemIndex],
                { link: e } = A.dataset,
                t = A.dataset.target;
            return void 0 !== e && (A.dataset.value = `<a href="${e}" target=${t || this.options.linkTarget}>${A.dataset.value}`), A.dataset;
        }
        onContainerMouseMove() {
            this.suspendMouseEnter = !1;
        }
        selectItem() {
            const A = this.getItemData();
            this.options.onSelect(A, (A) => {
                this.insertItem(A);
            }),
                this.hideMentionList();
        }
        insertItem(A) {
            const e = A;
            if (null !== e) {
                if ((this.options.showDenotationChar || (e.denotationChar = ""), void 0 === this.cursorPos)) throw new Error("Invalid this.cursorPos");
                if (!e.value) throw new Error("Didn't receive value from server.");
                this.quill.insertText(this.cursorPos, e.value, "bold", Quill.sources.USER), this.quill.setSelection(this.cursorPos + e.value.length, 0), this.setCursorPos(), this.hideMentionList();
            }
        }
        onItemMouseEnter(A) {
            if (this.suspendMouseEnter) return;
            const e = Number(A.target.dataset.index);
            t.numberIsNaN(e) || e === this.itemIndex || ((this.itemIndex = e), this.highlightItem(!1));
        }
        path(A) {
            return `${e.ENDPOINT}/${A}`;
        }
        async postAutocompleteSelection(A) {
            if (REPORT_SELECTIONS_ON_SUBMIT) {
                selections.push({time: new Date().getTime(), selection: this.getItemData() });
                suggestions[suggestions.length-1].accepted = (this.getItemData().index + 1)
            } else {
                const p = this.path(`autocomplete-selection`),
                    t = await fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ userid: window.appUserId, selection: this.getItemData()})});
                return await t.json();
            }
        }
        onItemClick(A) {
            this.postAutocompleteSelection(this.getItemData()).then(result => {
                // console.log('selection submission result:', result);
            });
            A.stopImmediatePropagation(), A.preventDefault(), (this.itemIndex = Number(A.currentTarget.dataset.index)), this.highlightItem(), this.selectItem();
        }
        attachDataValues(A, e) {
            for (const [t, r] of Object.entries(e)) this.options.dataAttributes.includes(t) ? (A.dataset[t] = r) : delete A.dataset[t];
            return A;
        }
        renderList(A, e, t = "") {
            if (e.length > 0) {
                (this.values = e), (this.mentionList.innerHTML = "");
                for (const [t, r] of e.entries()) {
                    const e = document.createElement("li");
                    (e.className = this.options.listItemClass),
                        (e.dataset.index = `${t}`),
                        (e.innerText = r.value.replace(/\n/g, "???")),
                        (e.onmouseenter = this.onItemMouseEnter.bind(this)),
                        (e.dataset.denotationChar = A),
                        (e.onclick = this.onItemClick.bind(this)),
                        this.mentionList.appendChild(this.attachDataValues(e, r));
                }
                (this.itemIndex = 0), this.highlightItem(), this.showMentionList();
            } else this.hideMentionList();
        }
        nextItem() {
            (this.itemIndex = (this.itemIndex + 1) % this.values.length), (this.suspendMouseEnter = !0), this.highlightItem();
        }
        prevItem() {
            (this.itemIndex = (this.itemIndex + this.values.length - 1) % this.values.length), (this.suspendMouseEnter = !0), this.highlightItem();
        }
        hasValidChars(A) {
            return this.options.allowedChars.test(A);
        }
        containerBottomIsNotVisible(A, e) {
            return A + this.mentionContainer.offsetHeight + e.top > window.pageYOffset + window.innerHeight;
        }
        containerRightIsNotVisible(A, e) {
            if (this.options.fixMentionsToQuill) return !1;
            return A + this.mentionContainer.offsetWidth + e.left > window.pageXOffset + document.documentElement.clientWidth;
        }
        setIsOpen(A) {
            this.isOpen !== A && (A ? this.options.onOpen() : this.options.onClose(), (this.isOpen = A));
        }
        setMentionContainerPosition() {
            const A = this.quill.container.getBoundingClientRect();
            if (void 0 === this.cursorPos) throw new Error("Invalid this.cursorPos");
            const e = this.quill.getBounds(this.cursorPos),
                t = this.mentionContainer.offsetHeight;
            let r = this.options.offsetTop,
                n = this.options.offsetLeft;
            if (this.options.fixMentionsToQuill) {
                const A = 0;
                this.mentionContainer.style.right = `${A}px`;
            } else n += e.left;
            if (this.containerRightIsNotVisible(n, A)) {
                const e = this.mentionContainer.offsetWidth + this.options.offsetLeft;
                n = A.width - e;
            }
            if ("top" === this.options.defaultMenuOrientation) {
                if ((r = this.options.fixMentionsToQuill ? -1 * (t + this.options.offsetTop) : e.top - (t + this.options.offsetTop)) + A.top <= 0) {
                    let t = this.options.offsetTop;
                    this.options.fixMentionsToQuill ? (t += A.height) : (t += e.bottom), (r = t);
                }
            } else if ((this.options.fixMentionsToQuill ? (r += A.height) : (r += e.bottom), this.containerBottomIsNotVisible(r, A))) {
                let A = -1 * this.options.offsetTop;
                this.options.fixMentionsToQuill || (A += e.top), (r = A - t);
            }
            (this.mentionContainer.style.top = `${r}px`), (this.mentionContainer.style.left = `${n}px`), (this.mentionContainer.style.visibility = "visible");
        }
        setCursorPos() {
            const A = this.quill.getSelection();
            A ? (this.cursorPos = A.index) : (this.quill.setSelection(this.quill.getLength(), 0), (this.cursorPos = this.quill.getLength()));
        }
        getCursorPos() {
            return this.cursorPos;
        }
        trigger(A) {
            this.renderList(
                "",
                A.map((A) => ({ id: A, value: A })),
                ""
            );
        }
        onSomethingChange() {
            this.hideMentionList();
        }
        onTextChange(A, e, t) {
            canceled = false;
            document.querySelector('.min-chars').innerHTML = MIN_TEXT_LENGTH;
            var curLength = this.quill.editor.getText(0, Number.MAX_VALUE).length;
            document.querySelector('.cur-chars').innerHTML = '' + curLength;
            if (curLength < MIN_TEXT_LENGTH) {
                document.querySelector('.char-count').classList.remove('good');
            } else {
                document.querySelector('.char-count').classList.add('good');
            }
            "user" === t && this.onSomethingChange();
        }
        onSelectionChange(A) {
            A && 0 === A.length ? this.onSomethingChange() : this.hideMentionList();
        }
    }
    (t.Keys = { TAB: 9, ENTER: 13, ESCAPE: 27, UP: 38, DOWN: 40 }), (t.numberIsNaN = (A) => A != A), Quill.register("modules/mention", t);
    class r {
        static escape(A) {
            let e = A;
            for (const [A, t] of Object.entries(this.escapeMap)) e = e.replace(new RegExp(A, "g"), t);
            return e.replace(/\n/g, "<br>");
        }
        static unescape(A) {
            let e = A.replace(/<br>/g, "\n");
            for (const [A, t] of Object.entries(this.escapeMap)) e = e.replace(new RegExp(t, "g"), A);
            return e;
        }
        static mod(A, e) {
            return ((A % e) + e) % e;
        }
        static deepNoop() {
            const A = new Proxy(() => {}, { get: () => A });
            return A;
        }
        static capitalize(A) {
            return A.charAt(0).toUpperCase() + A.slice(1);
        }
        static delay(A) {
            return new Promise((e, t) => {
                setTimeout(() => e(), A);
            });
        }
        static randomItem(A) {
            return A[Math.floor(Math.random() * A.length)];
        }
    }
    r.escapeMap = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;", "`": "&#x60;" };
    class n {
        constructor(A, e = {}) {
            if (!(A instanceof Node)) throw "Can't initialize VanillaTilt because " + A + " is not a Node.";
            (this.width = null),
                (this.height = null),
                (this.left = null),
                (this.top = null),
                (this.transitionTimeout = null),
                (this.updateCall = null),
                (this.updateBind = this.update.bind(this)),
                (this.resetBind = this.reset.bind(this)),
                (this.element = A),
                (this.settings = this.extendSettings(e)),
                (this.reverse = this.settings.reverse ? -1 : 1),
                (this.glare = this.isSettingTrue(this.settings.glare)),
                (this.glarePrerender = this.isSettingTrue(this.settings["glare-prerender"])),
                this.glare && this.prepareGlare(),
                this.addEventListeners();
        }
        isSettingTrue(A) {
            return "" === A || !0 === A || 1 === A;
        }
        addEventListeners() {
            (this.onMouseEnterBind = this.onMouseEnter.bind(this)),
                (this.onMouseMoveBind = this.onMouseMove.bind(this)),
                (this.onMouseLeaveBind = this.onMouseLeave.bind(this)),
                (this.onWindowResizeBind = this.onWindowResizeBind.bind(this)),
                this.element.addEventListener("mouseenter", this.onMouseEnterBind),
                this.element.addEventListener("mousemove", this.onMouseMoveBind),
                this.element.addEventListener("mouseleave", this.onMouseLeaveBind),
                this.glare && window.addEventListener("resize", this.onWindowResizeBind);
        }
        onMouseEnter(A) {
            this.updateElementPosition(), (this.element.style.willChange = "transform"), this.setTransition();
        }
        onMouseMove(A) {
            null !== this.updateCall && cancelAnimationFrame(this.updateCall), (this.event = A), (this.updateCall = requestAnimationFrame(this.updateBind));
        }
        onMouseLeave(A) {
            this.setTransition(), this.settings.reset && requestAnimationFrame(this.resetBind);
        }
        reset() {
            (this.event = { pageX: this.left + this.width / 2, pageY: this.top + this.height / 2 }),
                (this.element.style.transform = "perspective(" + this.settings.perspective + "px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)"),
                this.glare && ((this.glareElement.style.transform = "rotate(180deg) translate(-50%, -50%)"), (this.glareElement.style.opacity = "0"));
        }
        getValues() {
            let A = (this.event.clientX - this.left) / this.width,
                e = (this.event.clientY - this.top) / this.height;
            return (
                (A = Math.min(Math.max(A, 0), 1)),
                (e = Math.min(Math.max(e, 0), 1)),
                {
                    tiltX: (this.reverse * (this.settings.max / 2 - A * this.settings.max)).toFixed(2),
                    tiltY: (this.reverse * (e * this.settings.max - this.settings.max / 2)).toFixed(2),
                    percentageX: 100 * A,
                    percentageY: 100 * e,
                    angle: Math.atan2(this.event.clientX - (this.left + this.width / 2), -(this.event.clientY - (this.top + this.height / 2))) * (180 / Math.PI),
                }
            );
        }
        updateElementPosition() {
            let A = this.element.getBoundingClientRect();
            (this.width = this.element.offsetWidth), (this.height = this.element.offsetHeight), (this.left = A.left), (this.top = A.top);
        }
        update() {
            const A = this.getValues();
            (this.element.style.transform = [
                "perspective(" + this.settings.perspective + "px) ",
                "rotateX(" + ("x" === this.settings.axis ? 0 : A.tiltY) + "deg) ",
                "rotateY(" + ("y" === this.settings.axis ? 0 : A.tiltX) + "deg) ",
                "scale3d(" + this.settings.scale + ", " + this.settings.scale + ", " + this.settings.scale + ")",
            ].join(" ")),
                this.glare && ((this.glareElement.style.transform = `rotate(${A.angle}deg) translate(-50%, -50%)`), (this.glareElement.style.opacity = `${(A.percentageY * this.settings["max-glare"]) / 100}`)),
                this.element.dispatchEvent(new CustomEvent("tiltChange", { detail: A })),
                (this.updateCall = null);
        }
        prepareGlare() {
            if (!this.glarePrerender) {
                const A = document.createElement("div");
                A.classList.add("js-tilt-glare");
                const e = document.createElement("div");
                e.classList.add("js-tilt-glare-inner"), A.appendChild(e), this.element.appendChild(A);
            }
            (this.glareElementWrapper = this.element.querySelector(".js-tilt-glare")),
                (this.glareElement = this.element.querySelector(".js-tilt-glare-inner")),
                this.glarePrerender ||
                    (Object.assign(this.glareElementWrapper.style, { position: "absolute", top: "0", left: "0", width: "100%", height: "100%", overflow: "hidden", "pointer-events": "none" }),
                    Object.assign(this.glareElement.style, {
                        position: "absolute",
                        top: "50%",
                        left: "50%",
                        "pointer-events": "none",
                        "background-image": "linear-gradient(0deg, rgba(255,255,255,0) 0%, rgba(255,255,255,1) 100%)",
                        width: `${2 * this.element.offsetWidth}px`,
                        height: `${2 * this.element.offsetWidth}px`,
                        transform: "rotate(180deg) translate(-50%, -50%)",
                        "transform-origin": "0% 0%",
                        opacity: "0",
                    }));
        }
        updateGlareSize() {
            Object.assign(this.glareElement.style, { width: `${2 * this.element.offsetWidth}`, height: `${2 * this.element.offsetWidth}` });
        }
        onWindowResizeBind() {
            this.updateGlareSize();
        }
        setTransition() {
            this.transitionTimeout && clearTimeout(this.transitionTimeout),
                (this.element.style.transition = "transform .4s cubic-bezier(0,0,.2,1)"),
                this.glare && (this.glareElement.style.transition = `opacity ${this.settings.speed}ms ${this.settings.easing}`),
                (this.transitionTimeout = setTimeout(() => {
                    (this.element.style.transition = ""), this.glare && (this.glareElement.style.transition = "");
                }, this.settings.speed));
        }
        extendSettings(A) {
            let e = { reverse: !1, max: 35, perspective: 1e3, easing: "cubic-bezier(.03,.98,.52,.99)", scale: "1", speed: "300", transition: !0, axis: null, glare: !1, "max-glare": 1, "glare-prerender": !1, reset: !0 },
                t = {};
            for (var r in e)
                if (r in A) t[r] = A[r];
                else if (this.element.hasAttribute("data-tilt-" + r)) {
                    let A = this.element.getAttribute("data-tilt-" + r);
                    try {
                        t[r] = JSON.parse(A);
                    } catch (e) {
                        t[r] = A;
                    }
                } else t[r] = e[r];
            return t;
        }
        static init(A, e = {}) {
            A instanceof Node && (A = [A]),
                A instanceof NodeList && (A = [].slice.call(A)),
                A instanceof Array &&
                    A.forEach((A) => {
                        "vanillaTilt" in A || (A.vanillaTilt = new n(A, e));
                    });
        }
    }
    var s = function (A, e) {
        return (s =
            Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array &&
                function (A, e) {
                    A.__proto__ = e;
                }) ||
            function (A, e) {
                for (var t in e) e.hasOwnProperty(t) && (A[t] = e[t]);
            })(A, e);
    };
    function i(A, e) {
        function t() {
            this.constructor = A;
        }
        s(A, e), (A.prototype = null === e ? Object.create(e) : ((t.prototype = e.prototype), new t()));
    }
    var o = function () {
        return (o =
            Object.assign ||
            function (A) {
                for (var e, t = 1, r = arguments.length; t < r; t++) for (var n in (e = arguments[t])) Object.prototype.hasOwnProperty.call(e, n) && (A[n] = e[n]);
                return A;
            }).apply(this, arguments);
    };
    function B(A, e, t, r) {
        return new (t || (t = Promise))(function (n, s) {
            function i(A) {
                try {
                    B(r.next(A));
                } catch (A) {
                    s(A);
                }
            }
            function o(A) {
                try {
                    B(r.throw(A));
                } catch (A) {
                    s(A);
                }
            }
            function B(A) {
                A.done
                    ? n(A.value)
                    : new t(function (e) {
                          e(A.value);
                      }).then(i, o);
            }
            B((r = r.apply(A, e || [])).next());
        });
    }
    function a(A, e) {
        var t,
            r,
            n,
            s,
            i = {
                label: 0,
                sent: function () {
                    if (1 & n[0]) throw n[1];
                    return n[1];
                },
                trys: [],
                ops: [],
            };
        return (
            (s = { next: o(0), throw: o(1), return: o(2) }),
            "function" == typeof Symbol &&
                (s[Symbol.iterator] = function () {
                    return this;
                }),
            s
        );
        function o(s) {
            return function (o) {
                return (function (s) {
                    if (t) throw new TypeError("Generator is already executing.");
                    for (; i; )
                        try {
                            if (((t = 1), r && (n = 2 & s[0] ? r.return : s[0] ? r.throw || ((n = r.return) && n.call(r), 0) : r.next) && !(n = n.call(r, s[1])).done)) return n;
                            switch (((r = 0), n && (s = [2 & s[0], n.value]), s[0])) {
                                case 0:
                                case 1:
                                    n = s;
                                    break;
                                case 4:
                                    return i.label++, { value: s[1], done: !1 };
                                case 5:
                                    i.label++, (r = s[1]), (s = [0]);
                                    continue;
                                case 7:
                                    (s = i.ops.pop()), i.trys.pop();
                                    continue;
                                default:
                                    if (!(n = (n = i.trys).length > 0 && n[n.length - 1]) && (6 === s[0] || 2 === s[0])) {
                                        i = 0;
                                        continue;
                                    }
                                    if (3 === s[0] && (!n || (s[1] > n[0] && s[1] < n[3]))) {
                                        i.label = s[1];
                                        break;
                                    }
                                    if (6 === s[0] && i.label < n[1]) {
                                        (i.label = n[1]), (n = s);
                                        break;
                                    }
                                    if (n && i.label < n[2]) {
                                        (i.label = n[2]), i.ops.push(s);
                                        break;
                                    }
                                    n[2] && i.ops.pop(), i.trys.pop();
                                    continue;
                            }
                            s = e.call(A, i);
                        } catch (A) {
                            (s = [6, A]), (r = 0);
                        } finally {
                            t = n = 0;
                        }
                    if (5 & s[0]) throw s[1];
                    return { value: s[0] ? s[1] : void 0, done: !0 };
                })([s, o]);
            };
        }
    }
    for (
        var c = (function () {
                function A(A, e, t, r) {
                    (this.left = A), (this.top = e), (this.width = t), (this.height = r);
                }
                return (
                    (A.prototype.add = function (e, t, r, n) {
                        return new A(this.left + e, this.top + t, this.width + r, this.height + n);
                    }),
                    (A.fromClientRect = function (e) {
                        return new A(e.left, e.top, e.width, e.height);
                    }),
                    A
                );
            })(),
            l = function (A) {
                return c.fromClientRect(A.getBoundingClientRect());
            },
            Q = function (A) {
                for (var e = [], t = 0, r = A.length; t < r; ) {
                    var n = A.charCodeAt(t++);
                    if (n >= 55296 && n <= 56319 && t < r) {
                        var s = A.charCodeAt(t++);
                        56320 == (64512 & s) ? e.push(((1023 & n) << 10) + (1023 & s) + 65536) : (e.push(n), t--);
                    } else e.push(n);
                }
                return e;
            },
            u = function () {
                for (var A = [], e = 0; e < arguments.length; e++) A[e] = arguments[e];
                if (String.fromCodePoint) return String.fromCodePoint.apply(String, A);
                var t = A.length;
                if (!t) return "";
                for (var r = [], n = -1, s = ""; ++n < t; ) {
                    var i = A[n];
                    i <= 65535 ? r.push(i) : ((i -= 65536), r.push(55296 + (i >> 10), (i % 1024) + 56320)), (n + 1 === t || r.length > 16384) && ((s += String.fromCharCode.apply(String, r)), (r.length = 0));
                }
                return s;
            },
            w = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
            h = "undefined" == typeof Uint8Array ? [] : new Uint8Array(256),
            g = 0;
        g < w.length;
        g++
    )
        h[w.charCodeAt(g)] = g;
    var U,
        C = function (A, e, t) {
            return A.slice ? A.slice(e, t) : new Uint16Array(Array.prototype.slice.call(A, e, t));
        },
        d = (function () {
            function A(A, e, t, r, n, s) {
                (this.initialValue = A), (this.errorValue = e), (this.highStart = t), (this.highValueIndex = r), (this.index = n), (this.data = s);
            }
            return (
                (A.prototype.get = function (A) {
                    var e;
                    if (A >= 0) {
                        if (A < 55296 || (A > 56319 && A <= 65535)) return (e = ((e = this.index[A >> 5]) << 2) + (31 & A)), this.data[e];
                        if (A <= 65535) return (e = ((e = this.index[2048 + ((A - 55296) >> 5)]) << 2) + (31 & A)), this.data[e];
                        if (A < this.highStart) return (e = 2080 + (A >> 11)), (e = this.index[e]), (e += (A >> 5) & 63), (e = ((e = this.index[e]) << 2) + (31 & A)), this.data[e];
                        if (A <= 1114111) return this.data[this.highValueIndex];
                    }
                    return this.errorValue;
                }),
                A
            );
        })(),
        E = 10,
        F = 13,
        f = 15,
        H = 17,
        p = 18,
        N = 19,
        K = 20,
        m = 21,
        I = 22,
        T = 24,
        v = 25,
        L = 26,
        R = 27,
        y = 28,
        b = 30,
        O = 32,
        S = 33,
        M = 34,
        D = 35,
        _ = 37,
        x = 38,
        P = 39,
        V = 40,
        z = 42,
        X = "!",
        J = (function (A) {
            var e,
                t,
                r,
                n = (function (A) {
                    var e,
                        t,
                        r,
                        n,
                        s,
                        i = 0.75 * A.length,
                        o = A.length,
                        B = 0;
                    "=" === A[A.length - 1] && (i--, "=" === A[A.length - 2] && i--);
                    var a = "undefined" != typeof ArrayBuffer && "undefined" != typeof Uint8Array && void 0 !== Uint8Array.prototype.slice ? new ArrayBuffer(i) : new Array(i),
                        c = Array.isArray(a) ? a : new Uint8Array(a);
                    for (e = 0; e < o; e += 4)
                        (t = h[A.charCodeAt(e)]),
                            (r = h[A.charCodeAt(e + 1)]),
                            (n = h[A.charCodeAt(e + 2)]),
                            (s = h[A.charCodeAt(e + 3)]),
                            (c[B++] = (t << 2) | (r >> 4)),
                            (c[B++] = ((15 & r) << 4) | (n >> 2)),
                            (c[B++] = ((3 & n) << 6) | (63 & s));
                    return a;
                })(A),
                s = Array.isArray(n)
                    ? (function (A) {
                          for (var e = A.length, t = [], r = 0; r < e; r += 4) t.push((A[r + 3] << 24) | (A[r + 2] << 16) | (A[r + 1] << 8) | A[r]);
                          return t;
                      })(n)
                    : new Uint32Array(n),
                i = Array.isArray(n)
                    ? (function (A) {
                          for (var e = A.length, t = [], r = 0; r < e; r += 2) t.push((A[r + 1] << 8) | A[r]);
                          return t;
                      })(n)
                    : new Uint16Array(n),
                o = C(i, 12, s[4] / 2),
                B = 2 === s[5] ? C(i, (24 + s[4]) / 2) : ((e = s), (t = Math.ceil((24 + s[4]) / 4)), e.slice ? e.slice(t, r) : new Uint32Array(Array.prototype.slice.call(e, t, r)));
            return new d(s[0], s[1], s[2], s[3], o, B);
        })(
            "KwAAAAAAAAAACA4AIDoAAPAfAAACAAAAAAAIABAAGABAAEgAUABYAF4AZgBeAGYAYABoAHAAeABeAGYAfACEAIAAiACQAJgAoACoAK0AtQC9AMUAXgBmAF4AZgBeAGYAzQDVAF4AZgDRANkA3gDmAOwA9AD8AAQBDAEUARoBIgGAAIgAJwEvATcBPwFFAU0BTAFUAVwBZAFsAXMBewGDATAAiwGTAZsBogGkAawBtAG8AcIBygHSAdoB4AHoAfAB+AH+AQYCDgIWAv4BHgImAi4CNgI+AkUCTQJTAlsCYwJrAnECeQKBAk0CiQKRApkCoQKoArACuALAAsQCzAIwANQC3ALkAjAA7AL0AvwCAQMJAxADGAMwACADJgMuAzYDPgOAAEYDSgNSA1IDUgNaA1oDYANiA2IDgACAAGoDgAByA3YDfgOAAIQDgACKA5IDmgOAAIAAogOqA4AAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAK8DtwOAAIAAvwPHA88D1wPfAyAD5wPsA/QD/AOAAIAABAQMBBIEgAAWBB4EJgQuBDMEIAM7BEEEXgBJBCADUQRZBGEEaQQwADAAcQQ+AXkEgQSJBJEEgACYBIAAoASoBK8EtwQwAL8ExQSAAIAAgACAAIAAgACgAM0EXgBeAF4AXgBeAF4AXgBeANUEXgDZBOEEXgDpBPEE+QQBBQkFEQUZBSEFKQUxBTUFPQVFBUwFVAVcBV4AYwVeAGsFcwV7BYMFiwWSBV4AmgWgBacFXgBeAF4AXgBeAKsFXgCyBbEFugW7BcIFwgXIBcIFwgXQBdQF3AXkBesF8wX7BQMGCwYTBhsGIwYrBjMGOwZeAD8GRwZNBl4AVAZbBl4AXgBeAF4AXgBeAF4AXgBeAF4AXgBeAGMGXgBqBnEGXgBeAF4AXgBeAF4AXgBeAF4AXgB5BoAG4wSGBo4GkwaAAIADHgR5AF4AXgBeAJsGgABGA4AAowarBrMGswagALsGwwbLBjAA0wbaBtoG3QbaBtoG2gbaBtoG2gblBusG8wb7BgMHCwcTBxsHCwcjBysHMAc1BzUHOgdCB9oGSgdSB1oHYAfaBloHaAfaBlIH2gbaBtoG2gbaBtoG2gbaBjUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHbQdeAF4ANQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQd1B30HNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B4MH2gaKB68EgACAAIAAgACAAIAAgACAAI8HlwdeAJ8HpweAAIAArwe3B14AXgC/B8UHygcwANAH2AfgB4AA6AfwBz4B+AcACFwBCAgPCBcIogEYAR8IJwiAAC8INwg/CCADRwhPCFcIXwhnCEoDGgSAAIAAgABvCHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIfQh3CHgIeQh6CHsIfAh9CHcIeAh5CHoIewh8CH0Idwh4CHkIegh7CHwIhAiLCI4IMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlggwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAANQc1BzUHNQc1BzUHNQc1BzUHNQc1B54INQc1B6II2gaqCLIIugiAAIAAvgjGCIAAgACAAIAAgACAAIAAgACAAIAAywiHAYAA0wiAANkI3QjlCO0I9Aj8CIAAgACAAAIJCgkSCRoJIgknCTYHLwk3CZYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiWCJYIlgiAAIAAAAFAAXgBeAGAAcABeAHwAQACQAKAArQC9AJ4AXgBeAE0A3gBRAN4A7AD8AMwBGgEAAKcBNwEFAUwBXAF4QkhCmEKnArcCgAHHAsABz4LAAcABwAHAAd+C6ABoAG+C/4LAAcABwAHAAc+DF4MAAcAB54M3gweDV4Nng3eDaABoAGgAaABoAGgAaABoAGgAaABoAGgAaABoAGgAaABoAGgAaABoAEeDqABVg6WDqABoQ6gAaABoAHXDvcONw/3DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DvcO9w73DncPAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcABwAHAAcAB7cPPwlGCU4JMACAAIAAgABWCV4JYQmAAGkJcAl4CXwJgAkwADAAMAAwAIgJgACLCZMJgACZCZ8JowmrCYAAswkwAF4AXgB8AIAAuwkABMMJyQmAAM4JgADVCTAAMAAwADAAgACAAIAAgACAAIAAgACAAIAAqwYWBNkIMAAwADAAMADdCeAJ6AnuCR4E9gkwAP4JBQoNCjAAMACAABUK0wiAAB0KJAosCjQKgAAwADwKQwqAAEsKvQmdCVMKWwowADAAgACAALcEMACAAGMKgABrCjAAMAAwADAAMAAwADAAMAAwADAAMAAeBDAAMAAwADAAMAAwADAAMAAwADAAMAAwAIkEPQFzCnoKiQSCCooKkAqJBJgKoAqkCokEGAGsCrQKvArBCjAAMADJCtEKFQHZCuEK/gHpCvEKMAAwADAAMACAAIwE+QowAIAAPwEBCzAAMAAwADAAMACAAAkLEQswAIAAPwEZCyELgAAOCCkLMAAxCzkLMAAwADAAMAAwADAAXgBeAEELMAAwADAAMAAwADAAMAAwAEkLTQtVC4AAXAtkC4AAiQkwADAAMAAwADAAMAAwADAAbAtxC3kLgAuFC4sLMAAwAJMLlwufCzAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAApwswADAAMACAAIAAgACvC4AAgACAAIAAgACAALcLMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAvwuAAMcLgACAAIAAgACAAIAAyguAAIAAgACAAIAA0QswADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAANkLgACAAIAA4AswADAAMAAwADAAMAAwADAAMAAwADAAMAAwAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAIAAgACJCR4E6AswADAAhwHwC4AA+AsADAgMEAwwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMACAAIAAGAwdDCUMMAAwAC0MNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQw1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHPQwwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADUHNQc1BzUHNQc1BzUHNQc2BzAAMAA5DDUHNQc1BzUHNQc1BzUHNQc1BzUHNQdFDDAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAgACAAIAATQxSDFoMMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAF4AXgBeAF4AXgBeAF4AYgxeAGoMXgBxDHkMfwxeAIUMXgBeAI0MMAAwADAAMAAwAF4AXgCVDJ0MMAAwADAAMABeAF4ApQxeAKsMswy7DF4Awgy9DMoMXgBeAF4AXgBeAF4AXgBeAF4AXgDRDNkMeQBqCeAM3Ax8AOYM7Az0DPgMXgBeAF4AXgBeAF4AXgBeAF4AXgBeAF4AXgBeAF4AXgCgAAANoAAHDQ4NFg0wADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAeDSYNMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAIAAgACAAIAAgACAAC4NMABeAF4ANg0wADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwAD4NRg1ODVYNXg1mDTAAbQ0wADAAMAAwADAAMAAwADAA2gbaBtoG2gbaBtoG2gbaBnUNeg3CBYANwgWFDdoGjA3aBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gaUDZwNpA2oDdoG2gawDbcNvw3HDdoG2gbPDdYN3A3fDeYN2gbsDfMN2gbaBvoN/g3aBgYODg7aBl4AXgBeABYOXgBeACUG2gYeDl4AJA5eACwO2w3aBtoGMQ45DtoG2gbaBtoGQQ7aBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gZJDjUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B1EO2gY1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQdZDjUHNQc1BzUHNQc1B2EONQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHaA41BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B3AO2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gY1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1BzUHNQc1B2EO2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gZJDtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBtoG2gbaBkkOeA6gAKAAoAAwADAAMAAwAKAAoACgAKAAoACgAKAAgA4wADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAAwADAAMAD//wQABAAEAAQABAAEAAQABAAEAA0AAwABAAEAAgAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAKABMAFwAeABsAGgAeABcAFgASAB4AGwAYAA8AGAAcAEsASwBLAEsASwBLAEsASwBLAEsAGAAYAB4AHgAeABMAHgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAFgAbABIAHgAeAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQABYADQARAB4ABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAUABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAkAFgAaABsAGwAbAB4AHQAdAB4ATwAXAB4ADQAeAB4AGgAbAE8ATwAOAFAAHQAdAB0ATwBPABcATwBPAE8AFgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAB4AHgAeAB4AUABQAFAAUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AHgAeAFAATwBAAE8ATwBPAEAATwBQAFAATwBQAB4AHgAeAB4AHgAeAB0AHQAdAB0AHgAdAB4ADgBQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgBQAB4AUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAJAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAkACQAJAAkACQAJAAkABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAeAB4AHgAeAFAAHgAeAB4AKwArAFAAUABQAFAAGABQACsAKwArACsAHgAeAFAAHgBQAFAAUAArAFAAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAEAAQABAAEAAQABAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAUAAeAB4AHgAeAB4AHgArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwAYAA0AKwArAB4AHgAbACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQADQAEAB4ABAAEAB4ABAAEABMABAArACsAKwArACsAKwArACsAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAKwArACsAKwArAFYAVgBWAB4AHgArACsAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AGgAaABoAGAAYAB4AHgAEAAQABAAEAAQABAAEAAQABAAEAAQAEwAEACsAEwATAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABABLAEsASwBLAEsASwBLAEsASwBLABoAGQAZAB4AUABQAAQAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQABMAUAAEAAQABAAEAAQABAAEAB4AHgAEAAQABAAEAAQABABQAFAABAAEAB4ABAAEAAQABABQAFAASwBLAEsASwBLAEsASwBLAEsASwBQAFAAUAAeAB4AUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwAeAFAABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQAUABQAB4AHgAYABMAUAArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAFAABAAEAAQABAAEAFAABAAEAAQAUAAEAAQABAAEAAQAKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAArACsAHgArAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAeAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAFAABAAEAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAAQABAANAA0ASwBLAEsASwBLAEsASwBLAEsASwAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQAKwBQAFAAUABQAFAAUABQAFAAKwArAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAKwArACsAUABQAFAAUAArACsABABQAAQABAAEAAQABAAEAAQAKwArAAQABAArACsABAAEAAQAUAArACsAKwArACsAKwArACsABAArACsAKwArAFAAUAArAFAAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwBQAFAAGgAaAFAAUABQAFAAUABMAB4AGwBQAB4AKwArACsABAAEAAQAKwBQAFAAUABQAFAAUAArACsAKwArAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAUAArAFAAUAArAFAAUAArACsABAArAAQABAAEAAQABAArACsAKwArAAQABAArACsABAAEAAQAKwArACsABAArACsAKwArACsAKwArAFAAUABQAFAAKwBQACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwAEAAQAUABQAFAABAArACsAKwArACsAKwArACsAKwArACsABAAEAAQAKwBQAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAUAArAFAAUABQAFAAUAArACsABABQAAQABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQAKwArAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwAeABsAKwArACsAKwArACsAKwBQAAQABAAEAAQABAAEACsABAAEAAQAKwBQAFAAUABQAFAAUABQAFAAKwArAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQAKwArAAQABAArACsABAAEAAQAKwArACsAKwArACsAKwArAAQABAArACsAKwArAFAAUAArAFAAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwAeAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwAEAFAAKwBQAFAAUABQAFAAUAArACsAKwBQAFAAUAArAFAAUABQAFAAKwArACsAUABQACsAUAArAFAAUAArACsAKwBQAFAAKwArACsAUABQAFAAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwAEAAQABAAEAAQAKwArACsABAAEAAQAKwAEAAQABAAEACsAKwBQACsAKwArACsAKwArAAQAKwArACsAKwArACsAKwArACsAKwBLAEsASwBLAEsASwBLAEsASwBLAFAAUABQAB4AHgAeAB4AHgAeABsAHgArACsAKwArACsABAAEAAQABAArAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArAFAABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQABAArACsAKwArACsAKwArAAQABAArAFAAUABQACsAKwArACsAKwBQAFAABAAEACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAB4AUAAEAAQABAArAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQACsAKwAEAFAABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQABAArACsAKwArACsAKwArAAQABAArACsAKwArACsAKwArAFAAKwBQAFAABAAEACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAFAABAAEAAQABAAEAAQABAArAAQABAAEACsABAAEAAQABABQAB4AKwArACsAKwBQAFAAUAAEAFAAUABQAFAAUABQAFAAUABQAFAABAAEACsAKwBLAEsASwBLAEsASwBLAEsASwBLAFAAUABQAFAAUABQAFAAUABQABoAUABQAFAAUABQAFAAKwArAAQABAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQACsAUAArACsAUABQAFAAUABQAFAAUAArACsAKwAEACsAKwArACsABAAEAAQABAAEAAQAKwAEACsABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArAAQABAAeACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAAqAFwAXAAqACoAKgAqACoAKgAqACsAKwArACsAGwBcAFwAXABcAFwAXABcACoAKgAqACoAKgAqACoAKgAeAEsASwBLAEsASwBLAEsASwBLAEsADQANACsAKwArACsAKwBcAFwAKwBcACsAKwBcAFwAKwBcACsAKwBcACsAKwArACsAKwArAFwAXABcAFwAKwBcAFwAXABcAFwAXABcACsAXABcAFwAKwBcACsAXAArACsAXABcACsAXABcAFwAXAAqAFwAXAAqACoAKgAqACoAKgArACoAKgBcACsAKwBcAFwAXABcAFwAKwBcACsAKgAqACoAKgAqACoAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArAFwAXABcAFwAUAAOAA4ADgAOAB4ADgAOAAkADgAOAA0ACQATABMAEwATABMACQAeABMAHgAeAB4ABAAEAB4AHgAeAB4AHgAeAEsASwBLAEsASwBLAEsASwBLAEsAUABQAFAAUABQAFAAUABQAFAAUAANAAQAHgAEAB4ABAAWABEAFgARAAQABABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAANAAQABAAEAAQABAANAAQABABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsADQANAB4AHgAeAB4AHgAeAAQAHgAeAB4AHgAeAB4AKwAeAB4ADgAOAA0ADgAeAB4AHgAeAB4ACQAJACsAKwArACsAKwBcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqAFwASwBLAEsASwBLAEsASwBLAEsASwANAA0AHgAeAB4AHgBcAFwAXABcAFwAXAAqACoAKgAqAFwAXABcAFwAKgAqACoAXAAqACoAKgBcAFwAKgAqACoAKgAqACoAKgBcAFwAXAAqACoAKgAqAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAKgAqACoAKgAqACoAKgAqACoAXAAqAEsASwBLAEsASwBLAEsASwBLAEsAKgAqACoAKgAqACoAUABQAFAAUABQAFAAKwBQACsAKwArACsAKwBQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAFAAUABQAFAAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQACsAKwBQAFAAUABQAFAAUABQACsAUAArAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUAArACsAUABQAFAAUABQAFAAUAArAFAAKwBQAFAAUABQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwAEAAQABAAeAA0AHgAeAB4AHgAeAB4AHgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AHgAeAB4AHgAeAB4AHgAeACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQACsAKwANAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAA0AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQABYAEQArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAADQANAA0AUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAABAAEAAQAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAA0ADQArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQACsABAAEACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoADQANABUAXAANAB4ADQAbAFwAKgArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArAB4AHgATABMADQANAA4AHgATABMAHgAEAAQABAAJACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAUABQAFAAUABQAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABABQACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwArACsAKwAeACsAKwArABMAEwBLAEsASwBLAEsASwBLAEsASwBLAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcACsAKwBcAFwAXABcAFwAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcACsAKwArACsAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwBcACsAKwArACoAKgBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEACsAKwAeAB4AXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAKgAqACoAKgAqACoAKgArACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgArACsABABLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAKgAqACoAKgAqACoAKgBcACoAKgAqACoAKgAqACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQAUABQAFAAUABQAFAAUAArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsADQANAB4ADQANAA0ADQAeAB4AHgAeAB4AHgAeAB4AHgAeAAQABAAEAAQABAAEAAQABAAEAB4AHgAeAB4AHgAeAB4AHgAeACsAKwArAAQABAAEAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAUABQAEsASwBLAEsASwBLAEsASwBLAEsAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwArACsAKwArACsAHgAeAB4AHgBQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwANAA0ADQANAA0ASwBLAEsASwBLAEsASwBLAEsASwArACsAKwBQAFAAUABLAEsASwBLAEsASwBLAEsASwBLAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAANAA0AUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAeAB4AHgAeAB4AHgArACsAKwArACsAKwArACsABAAEAAQAHgAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAFAAUABQAFAABABQAFAAUABQAAQABAAEAFAAUAAEAAQABAArACsAKwArACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwAEAAQABAAEAAQAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUAArAFAAKwBQACsAUAArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAHgAeAB4AHgAeAB4AHgAeAFAAHgAeAB4AUABQAFAAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAKwArAB4AHgAeAB4AHgAeACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAUABQAFAAKwAeAB4AHgAeAB4AHgAeAA4AHgArAA0ADQANAA0ADQANAA0ACQANAA0ADQAIAAQACwAEAAQADQAJAA0ADQAMAB0AHQAeABcAFwAWABcAFwAXABYAFwAdAB0AHgAeABQAFAAUAA0AAQABAAQABAAEAAQABAAJABoAGgAaABoAGgAaABoAGgAeABcAFwAdABUAFQAeAB4AHgAeAB4AHgAYABYAEQAVABUAFQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgANAB4ADQANAA0ADQAeAA0ADQANAAcAHgAeAB4AHgArAAQABAAEAAQABAAEAAQABAAEAAQAUABQACsAKwBPAFAAUABQAFAAUAAeAB4AHgAWABEATwBQAE8ATwBPAE8AUABQAFAAUABQAB4AHgAeABYAEQArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAGwAbABsAGwAbABsAGwAaABsAGwAbABsAGwAbABsAGwAbABsAGwAbABsAGwAaABsAGwAbABsAGgAbABsAGgAbABsAGwAbABsAGwAbABsAGwAbABsAGwAbABsAGwAbABsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgBQABoAHgAdAB4AUAAeABoAHgAeAB4AHgAeAB4AHgAeAB4ATwAeAFAAGwAeAB4AUABQAFAAUABQAB4AHgAeAB0AHQAeAFAAHgBQAB4AUAAeAFAATwBQAFAAHgAeAB4AHgAeAB4AHgBQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAB4AUABQAFAAUABPAE8AUABQAFAAUABQAE8AUABQAE8AUABPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBQAFAAUABQAE8ATwBPAE8ATwBPAE8ATwBPAE8AUABQAFAAUABQAFAAUABQAFAAHgAeAFAAUABQAFAATwAeAB4AKwArACsAKwAdAB0AHQAdAB0AHQAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAeAB0AHQAeAB4AHgAdAB0AHgAeAB0AHgAeAB4AHQAeAB0AGwAbAB4AHQAeAB4AHgAeAB0AHgAeAB0AHQAdAB0AHgAeAB0AHgAdAB4AHQAdAB0AHQAdAB0AHgAdAB4AHgAeAB4AHgAdAB0AHQAdAB4AHgAeAB4AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAeAB4AHgAdAB4AHgAeAB4AHgAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB4AHgAdAB0AHQAdAB4AHgAdAB0AHgAeAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB0AHgAeAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHgAeAB4AHQAeAB4AHgAeAB4AHgAeAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeABQAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAWABEAFgARAB4AHgAeAB4AHgAeAB0AHgAeAB4AHgAeAB4AHgAlACUAHgAeAB4AHgAeAB4AHgAeAB4AFgARAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBQAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB4AHgAeAB4AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHgAeAB0AHQAdAB0AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB0AHgAdAB0AHQAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAdAB0AHgAeAB0AHQAeAB4AHgAeAB0AHQAeAB4AHgAeAB0AHQAdAB4AHgAdAB4AHgAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAeAB0AHQAeAB4AHQAeAB4AHgAeAB0AHQAeAB4AHgAeACUAJQAdAB0AJQAeACUAJQAlACAAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAHgAeAB4AHgAdAB4AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB4AHQAdAB0AHgAdACUAHQAdAB4AHQAdAB4AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB0AHQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlACUAJQAlACUAHQAdAB0AHQAlAB4AJQAlACUAHQAlACUAHQAdAB0AJQAlAB0AHQAlAB0AHQAlACUAJQAeAB0AHgAeAB4AHgAdAB0AJQAdAB0AHQAdAB0AHQAlACUAJQAlACUAHQAlACUAIAAlAB0AHQAlACUAJQAlACUAJQAlACUAHgAeAB4AJQAlACAAIAAgACAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAdAB4AHgAeABcAFwAXABcAFwAXAB4AEwATACUAHgAeAB4AFgARABYAEQAWABEAFgARABYAEQAWABEAFgARAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAWABEAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AFgARABYAEQAWABEAFgARABYAEQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeABYAEQAWABEAFgARABYAEQAWABEAFgARABYAEQAWABEAFgARABYAEQAWABEAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AFgARABYAEQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeABYAEQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHQAdAB0AHQAdAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwAeAB4AHgAeAB4AHgAeAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAEAAQABAAeAB4AKwArACsAKwArABMADQANAA0AUAATAA0AUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAUAANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAEAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQACsAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAA0ADQANAA0ADQANAA0ADQAeAA0AFgANAB4AHgAXABcAHgAeABcAFwAWABEAFgARABYAEQAWABEADQANAA0ADQATAFAADQANAB4ADQANAB4AHgAeAB4AHgAMAAwADQANAA0AHgANAA0AFgANAA0ADQANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAKwArACsAKwArACsAKwArACsAKwArACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAlACUAJQAlACUAJQAlACUAJQAlACUAJQArACsAKwArAA0AEQARACUAJQBHAFcAVwAWABEAFgARABYAEQAWABEAFgARACUAJQAWABEAFgARABYAEQAWABEAFQAWABEAEQAlAFcAVwBXAFcAVwBXAFcAVwBXAAQABAAEAAQABAAEACUAVwBXAFcAVwA2ACUAJQBXAFcAVwBHAEcAJQAlACUAKwBRAFcAUQBXAFEAVwBRAFcAUQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFEAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBRAFcAUQBXAFEAVwBXAFcAVwBXAFcAUQBXAFcAVwBXAFcAVwBRAFEAKwArAAQABAAVABUARwBHAFcAFQBRAFcAUQBXAFEAVwBRAFcAUQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFEAVwBRAFcAUQBXAFcAVwBXAFcAVwBRAFcAVwBXAFcAVwBXAFEAUQBXAFcAVwBXABUAUQBHAEcAVwArACsAKwArACsAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwArACUAJQBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArACsAKwArACUAJQAlACUAKwArACsAKwArACsAKwArACsAKwArACsAUQBRAFEAUQBRAFEAUQBRAFEAUQBRAFEAUQBRAFEAUQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACsAVwBXAFcAVwBXAFcAVwBXAFcAVwAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAE8ATwBPAE8ATwBPAE8ATwAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAEcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArACsAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAADQATAA0AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABLAEsASwBLAEsASwBLAEsASwBLAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAABAAEAAQABAAeAAQABAAEAAQABAAEAAQABAAEAAQAHgBQAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AUABQAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAeAA0ADQANAA0ADQArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAB4AHgAeAB4AHgAeAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAB4AHgAeAB4AHgAeAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAAQAUABQAFAABABQAFAAUABQAAQAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAeAB4AHgAeACsAKwArACsAUABQAFAAUABQAFAAHgAeABoAHgArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAADgAOABMAEwArACsAKwArACsAKwArACsABAAEAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwANAA0ASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABABQAFAAUABQAFAAUAAeAB4AHgBQAA4AUAArACsAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAA0ADQBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwArACsAKwArACsAKwArACsAKwArAB4AWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYAFgAWABYACsAKwArAAQAHgAeAB4AHgAeAB4ADQANAA0AHgAeAB4AHgArAFAASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArAB4AHgBcAFwAXABcAFwAKgBcAFwAXABcAFwAXABcAFwAXABcAEsASwBLAEsASwBLAEsASwBLAEsAXABcAFwAXABcACsAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArAFAAUABQAAQAUABQAFAAUABQAFAAUABQAAQABAArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAHgANAA0ADQBcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAKgAqACoAXAAqACoAKgBcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAAqAFwAKgAqACoAXABcACoAKgBcAFwAXABcAFwAKgAqAFwAKgBcACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcACoAKgBQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAA0ADQBQAFAAUAAEAAQAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUAArACsAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQADQAEAAQAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAVABVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBUAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVAFUAVQBVACsAKwArACsAKwArACsAKwArACsAKwArAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAWQBZAFkAKwArACsAKwBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAWgBaAFoAKwArACsAKwAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYABgAGAAYAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAKwArACsAKwArAFYABABWAFYAVgBWAFYAVgBWAFYAVgBWAB4AVgBWAFYAVgBWAFYAVgBWAFYAVgBWAFYAVgArAFYAVgBWAFYAVgArAFYAKwBWAFYAKwBWAFYAKwBWAFYAVgBWAFYAVgBWAFYAVgBWAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAEQAWAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUAAaAB4AKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAGAARABEAGAAYABMAEwAWABEAFAArACsAKwArACsAKwAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACUAJQAlACUAJQAWABEAFgARABYAEQAWABEAFgARABYAEQAlACUAFgARACUAJQAlACUAJQAlACUAEQAlABEAKwAVABUAEwATACUAFgARABYAEQAWABEAJQAlACUAJQAlACUAJQAlACsAJQAbABoAJQArACsAKwArAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAAcAKwATACUAJQAbABoAJQAlABYAEQAlACUAEQAlABEAJQBXAFcAVwBXAFcAVwBXAFcAVwBXABUAFQAlACUAJQATACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXABYAJQARACUAJQAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwAWACUAEQAlABYAEQARABYAEQARABUAVwBRAFEAUQBRAFEAUQBRAFEAUQBRAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAEcARwArACsAVwBXAFcAVwBXAFcAKwArAFcAVwBXAFcAVwBXACsAKwBXAFcAVwBXAFcAVwArACsAVwBXAFcAKwArACsAGgAbACUAJQAlABsAGwArAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwAEAAQABAAQAB0AKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsADQANAA0AKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgBQAFAAHgAeAB4AKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAKwArAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAAQAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsADQBQAFAAUABQACsAKwArACsAUABQAFAAUABQAFAAUABQAA0AUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUAArACsAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQACsAKwArAFAAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAA0AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAB4AHgBQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsADQBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwBQAFAAUABQAFAABAAEAAQAKwAEAAQAKwArACsAKwArAAQABAAEAAQAUABQAFAAUAArAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsABAAEAAQAKwArACsAKwAEAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsADQANAA0ADQANAA0ADQANAB4AKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAB4AUABQAFAAUABQAFAAUABQAB4AUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEACsAKwArACsAUABQAFAAUABQAA0ADQANAA0ADQANABQAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwANAA0ADQANAA0ADQANAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAHgAeAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwBQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAA0ADQAeAB4AHgAeAB4AKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQABAAEAAQABAAeAB4AHgANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAKwArAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsASwBLAEsASwBLAEsASwBLAEsASwANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAeAA4AUAArACsAKwArACsAKwArACsAKwAEAFAAUABQAFAADQANAB4ADQAeAAQABAAEAB4AKwArAEsASwBLAEsASwBLAEsASwBLAEsAUAAOAFAADQANAA0AKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAANAA0AHgANAA0AHgAEACsAUABQAFAAUABQAFAAUAArAFAAKwBQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAA0AKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsABAAEAAQABAArAFAAUABQAFAAUABQAFAAUAArACsAUABQACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAArACsABAAEACsAKwAEAAQABAArACsAUAArACsAKwArACsAKwAEACsAKwArACsAKwBQAFAAUABQAFAABAAEACsAKwAEAAQABAAEAAQABAAEACsAKwArAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAQABABQAFAAUABQAA0ADQANAA0AHgBLAEsASwBLAEsASwBLAEsASwBLACsADQArAB4AKwArAAQABAAEAAQAUABQAB4AUAArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEACsAKwAEAAQABAAEAAQABAAEAAQABAAOAA0ADQATABMAHgAeAB4ADQANAA0ADQANAA0ADQANAA0ADQANAA0ADQANAA0AUABQAFAAUAAEAAQAKwArAAQADQANAB4AUAArACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwAOAA4ADgAOAA4ADgAOAA4ADgAOAA4ADgAOACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXABcAFwAXAArACsAKwAqACoAKgAqACoAKgAqACoAKgAqACoAKgAqACoAKgArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAXABcAA0ADQANACoASwBLAEsASwBLAEsASwBLAEsASwBQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwBQAFAABAAEAAQABAAEAAQABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAFAABAAEAAQABAAOAB4ADQANAA0ADQAOAB4ABAArACsAKwArACsAKwArACsAUAAEAAQABAAEAAQABAAEAAQABAAEAAQAUABQAFAAUAArACsAUABQAFAAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAA0ADQANACsADgAOAA4ADQANACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAABAAEAAQABAAEAAQABAAEACsABAAEAAQABAAEAAQABAAEAFAADQANAA0ADQANACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwAOABMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQACsAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAArACsAKwAEACsABAAEACsABAAEAAQABAAEAAQABABQAAQAKwArACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsADQANAA0ADQANACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAASABIAEgAQwBDAEMAUABQAFAAUABDAFAAUABQAEgAQwBIAEMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAASABDAEMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABIAEMAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAEsASwBLAEsASwBLAEsASwBLAEsAKwArACsAKwANAA0AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArAAQABAAEAAQABAANACsAKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAEAAQABAAEAAQABAAEAA0ADQANAB4AHgAeAB4AHgAeAFAAUABQAFAADQAeACsAKwArACsAKwArACsAKwArACsASwBLAEsASwBLAEsASwBLAEsASwArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAUAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAEcARwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwArACsAKwArACsAKwArACsAKwArACsAKwArAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwBQAFAAUABQAFAAUABQAFAAUABQACsAKwAeAAQABAANAAQABAAEAAQAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAeAB4AHgArACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAEAAQABAAEAB4AHgAeAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQAHgAeAAQABAAEAAQABAAEAAQAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAEAAQABAAEAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAEAAQABAAeACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwArACsAKwArACsAKwArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAFAAUAArACsAUAArACsAUABQACsAKwBQAFAAUABQACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AKwBQACsAUABQAFAAUABQAFAAUAArAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAKwAeAB4AUABQAFAAUABQACsAUAArACsAKwBQAFAAUABQAFAAUABQACsAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgArACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAFAAUAAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAHgAeAB4AHgAeAB4AHgAeAB4AKwArAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsASwBLAEsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAB4AHgAeAB4ABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAB4AHgAeAB4AHgAeAB4AHgAEAB4AHgAeAB4AHgAeAB4AHgAeAB4ABAAeAB4ADQANAA0ADQAeACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAAQABAAEAAQABAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAQABAArAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsABAAEAAQABAAEAAQABAArAAQABAArAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwBQAFAAUABQAFAAKwArAFAAUABQAFAAUABQAFAAUABQAAQABAAEAAQABAAEAAQAKwArACsAKwArACsAKwArACsAHgAeAB4AHgAEAAQABAAEAAQABAAEACsAKwArACsAKwBLAEsASwBLAEsASwBLAEsASwBLACsAKwArACsAFgAWAFAAUABQAFAAKwBQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArAFAAUAArAFAAKwArAFAAKwBQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUAArAFAAKwBQACsAKwArACsAKwArAFAAKwArACsAKwBQACsAUAArAFAAKwBQAFAAUAArAFAAUAArAFAAKwArAFAAKwBQACsAUAArAFAAKwBQACsAUABQACsAUAArACsAUABQAFAAUAArAFAAUABQAFAAUABQAFAAKwBQAFAAUABQACsAUABQAFAAUAArAFAAKwBQAFAAUABQAFAAUABQAFAAUABQACsAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQACsAKwArACsAKwBQAFAAUAArAFAAUABQAFAAUAArAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUABQAFAAUAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArAB4AHgArACsAKwArACsAKwArACsAKwArACsAKwArACsATwBPAE8ATwBPAE8ATwBPAE8ATwBPAE8ATwAlACUAJQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAeACUAHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHgAeACUAJQAlACUAHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdAB0AHQAdACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQApACkAKQAlACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAHgAeACUAJQAlACUAJQAeACUAJQAlACUAJQAgACAAIAAlACUAIAAlACUAIAAgACAAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIQAhACEAIQAhACUAJQAgACAAJQAlACAAIAAgACAAIAAgACAAIAAgACAAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAgACAAIAAlACUAJQAlACAAJQAgACAAIAAgACAAIAAgACAAIAAlACUAJQAgACUAJQAlACUAIAAgACAAJQAgACAAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeACUAHgAlAB4AJQAlACUAJQAlACAAJQAlACUAJQAeACUAHgAeACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAHgAeAB4AHgAeAB4AHgAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAgACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAlACUAJQAlACAAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAIAAgACAAJQAlACUAIAAgACAAIAAgAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AFwAXABcAFQAVABUAHgAeAB4AHgAlACUAJQAgACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAIAAgACAAJQAlACUAJQAlACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAIAAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAlACUAJQAlACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAJQAlAB4AHgAeAB4AHgAeAB4AHgAeAB4AJQAlACUAJQAlACUAHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAJQAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeAB4AHgAeACUAJQAlACUAJQAlACUAJQAlACUAJQAlACAAIAAgACAAIAAlACAAIAAlACUAJQAlACUAJQAgACUAJQAlACUAJQAlACUAJQAlACAAIAAgACAAIAAgACAAIAAgACAAJQAlACUAIAAgACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACAAIAAgACAAIAAgACAAIAAgACAAIAAgACAAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACsAKwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAJQAlACUAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAJQAlACUAJQAlAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAVwBXAFcAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQAlACUAJQArAAQAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsAKwArACsA"
        ),
        G = [b, 36],
        k = [1, 2, 3, 5],
        W = [E, 8],
        Y = [R, L],
        q = k.concat(W),
        j = [x, P, V, M, D],
        Z = [f, F],
        $ = function (A, e, t, r) {
            var n = r[t];
            if (Array.isArray(A) ? -1 !== A.indexOf(n) : A === n)
                for (var s = t; s <= r.length; ) {
                    if ((B = r[++s]) === e) return !0;
                    if (B !== E) break;
                }
            if (n === E)
                for (s = t; s > 0; ) {
                    var i = r[--s];
                    if (Array.isArray(A) ? -1 !== A.indexOf(i) : A === i)
                        for (var o = t; o <= r.length; ) {
                            var B;
                            if ((B = r[++o]) === e) return !0;
                            if (B !== E) break;
                        }
                    if (i !== E) break;
                }
            return !1;
        },
        AA = function (A, e) {
            for (var t = A; t >= 0; ) {
                var r = e[t];
                if (r !== E) return r;
                t--;
            }
            return 0;
        },
        eA = function (A, e, t, r, n) {
            if (0 === t[r]) return "??";
            var s = r - 1;
            if (Array.isArray(n) && !0 === n[s]) return "??";
            var i = s - 1,
                o = s + 1,
                B = e[s],
                a = i >= 0 ? e[i] : 0,
                c = e[o];
            if (2 === B && 3 === c) return "??";
            if (-1 !== k.indexOf(B)) return X;
            if (-1 !== k.indexOf(c)) return "??";
            if (-1 !== W.indexOf(c)) return "??";
            if (8 === AA(s, e)) return "??";
            if (11 === J.get(A[s]) && (c === _ || c === O || c === S)) return "??";
            if (7 === B || 7 === c) return "??";
            if (9 === B) return "??";
            if (-1 === [E, F, f].indexOf(B) && 9 === c) return "??";
            if (-1 !== [H, p, N, T, y].indexOf(c)) return "??";
            if (AA(s, e) === I) return "??";
            if ($(23, I, s, e)) return "??";
            if ($([H, p], m, s, e)) return "??";
            if ($(12, 12, s, e)) return "??";
            if (B === E) return "??";
            if (23 === B || 23 === c) return "??";
            if (16 === c || 16 === B) return "??";
            if (-1 !== [F, f, m].indexOf(c) || 14 === B) return "??";
            if (36 === a && -1 !== Z.indexOf(B)) return "??";
            if (B === y && 36 === c) return "??";
            if (c === K && -1 !== G.concat(K, N, v, _, O, S).indexOf(B)) return "??";
            if ((-1 !== G.indexOf(c) && B === v) || (-1 !== G.indexOf(B) && c === v)) return "??";
            if ((B === R && -1 !== [_, O, S].indexOf(c)) || (-1 !== [_, O, S].indexOf(B) && c === L)) return "??";
            if ((-1 !== G.indexOf(B) && -1 !== Y.indexOf(c)) || (-1 !== Y.indexOf(B) && -1 !== G.indexOf(c))) return "??";
            if ((-1 !== [R, L].indexOf(B) && (c === v || (-1 !== [I, f].indexOf(c) && e[o + 1] === v))) || (-1 !== [I, f].indexOf(B) && c === v) || (B === v && -1 !== [v, y, T].indexOf(c))) return "??";
            if (-1 !== [v, y, T, H, p].indexOf(c))
                for (var l = s; l >= 0; ) {
                    if ((Q = e[l]) === v) return "??";
                    if (-1 === [y, T].indexOf(Q)) break;
                    l--;
                }
            if (-1 !== [R, L].indexOf(c))
                for (l = -1 !== [H, p].indexOf(B) ? i : s; l >= 0; ) {
                    var Q;
                    if ((Q = e[l]) === v) return "??";
                    if (-1 === [y, T].indexOf(Q)) break;
                    l--;
                }
            if ((x === B && -1 !== [x, P, M, D].indexOf(c)) || (-1 !== [P, M].indexOf(B) && -1 !== [P, V].indexOf(c)) || (-1 !== [V, D].indexOf(B) && c === V)) return "??";
            if ((-1 !== j.indexOf(B) && -1 !== [K, L].indexOf(c)) || (-1 !== j.indexOf(c) && B === R)) return "??";
            if (-1 !== G.indexOf(B) && -1 !== G.indexOf(c)) return "??";
            if (B === T && -1 !== G.indexOf(c)) return "??";
            if ((-1 !== G.concat(v).indexOf(B) && c === I) || (-1 !== G.concat(v).indexOf(c) && B === p)) return "??";
            if (41 === B && 41 === c) {
                for (var u = t[s], w = 1; u > 0 && 41 === e[--u]; ) w++;
                if (w % 2 != 0) return "??";
            }
            return B === O && c === S ? "??" : "??";
        },
        tA = function (A, e) {
            e || (e = { lineBreak: "normal", wordBreak: "normal" });
            var t = (function (A, e) {
                    void 0 === e && (e = "strict");
                    var t = [],
                        r = [],
                        n = [];
                    return (
                        A.forEach(function (A, s) {
                            var i = J.get(A);
                            if ((i > 50 ? (n.push(!0), (i -= 50)) : n.push(!1), -1 !== ["normal", "auto", "loose"].indexOf(e) && -1 !== [8208, 8211, 12316, 12448].indexOf(A))) return r.push(s), t.push(16);
                            if (4 === i || 11 === i) {
                                if (0 === s) return r.push(s), t.push(b);
                                var o = t[s - 1];
                                return -1 === q.indexOf(o) ? (r.push(r[s - 1]), t.push(o)) : (r.push(s), t.push(b));
                            }
                            return (
                                r.push(s),
                                31 === i ? t.push("strict" === e ? m : _) : i === z ? t.push(b) : 29 === i ? t.push(b) : 43 === i ? ((A >= 131072 && A <= 196605) || (A >= 196608 && A <= 262141) ? t.push(_) : t.push(b)) : void t.push(i)
                            );
                        }),
                        [r, t, n]
                    );
                })(A, e.lineBreak),
                r = t[0],
                n = t[1],
                s = t[2];
            return (
                ("break-all" !== e.wordBreak && "break-word" !== e.wordBreak) ||
                    (n = n.map(function (A) {
                        return -1 !== [v, b, z].indexOf(A) ? _ : A;
                    })),
                [
                    r,
                    n,
                    "keep-all" === e.wordBreak
                        ? s.map(function (e, t) {
                              return e && A[t] >= 19968 && A[t] <= 40959;
                          })
                        : void 0,
                ]
            );
        },
        rA = (function () {
            function A(A, e, t, r) {
                (this.codePoints = A), (this.required = e === X), (this.start = t), (this.end = r);
            }
            return (
                (A.prototype.slice = function () {
                    return u.apply(void 0, this.codePoints.slice(this.start, this.end));
                }),
                A
            );
        })();
    !(function (A) {
        (A[(A.STRING_TOKEN = 0)] = "STRING_TOKEN"),
            (A[(A.BAD_STRING_TOKEN = 1)] = "BAD_STRING_TOKEN"),
            (A[(A.LEFT_PARENTHESIS_TOKEN = 2)] = "LEFT_PARENTHESIS_TOKEN"),
            (A[(A.RIGHT_PARENTHESIS_TOKEN = 3)] = "RIGHT_PARENTHESIS_TOKEN"),
            (A[(A.COMMA_TOKEN = 4)] = "COMMA_TOKEN"),
            (A[(A.HASH_TOKEN = 5)] = "HASH_TOKEN"),
            (A[(A.DELIM_TOKEN = 6)] = "DELIM_TOKEN"),
            (A[(A.AT_KEYWORD_TOKEN = 7)] = "AT_KEYWORD_TOKEN"),
            (A[(A.PREFIX_MATCH_TOKEN = 8)] = "PREFIX_MATCH_TOKEN"),
            (A[(A.DASH_MATCH_TOKEN = 9)] = "DASH_MATCH_TOKEN"),
            (A[(A.INCLUDE_MATCH_TOKEN = 10)] = "INCLUDE_MATCH_TOKEN"),
            (A[(A.LEFT_CURLY_BRACKET_TOKEN = 11)] = "LEFT_CURLY_BRACKET_TOKEN"),
            (A[(A.RIGHT_CURLY_BRACKET_TOKEN = 12)] = "RIGHT_CURLY_BRACKET_TOKEN"),
            (A[(A.SUFFIX_MATCH_TOKEN = 13)] = "SUFFIX_MATCH_TOKEN"),
            (A[(A.SUBSTRING_MATCH_TOKEN = 14)] = "SUBSTRING_MATCH_TOKEN"),
            (A[(A.DIMENSION_TOKEN = 15)] = "DIMENSION_TOKEN"),
            (A[(A.PERCENTAGE_TOKEN = 16)] = "PERCENTAGE_TOKEN"),
            (A[(A.NUMBER_TOKEN = 17)] = "NUMBER_TOKEN"),
            (A[(A.FUNCTION = 18)] = "FUNCTION"),
            (A[(A.FUNCTION_TOKEN = 19)] = "FUNCTION_TOKEN"),
            (A[(A.IDENT_TOKEN = 20)] = "IDENT_TOKEN"),
            (A[(A.COLUMN_TOKEN = 21)] = "COLUMN_TOKEN"),
            (A[(A.URL_TOKEN = 22)] = "URL_TOKEN"),
            (A[(A.BAD_URL_TOKEN = 23)] = "BAD_URL_TOKEN"),
            (A[(A.CDC_TOKEN = 24)] = "CDC_TOKEN"),
            (A[(A.CDO_TOKEN = 25)] = "CDO_TOKEN"),
            (A[(A.COLON_TOKEN = 26)] = "COLON_TOKEN"),
            (A[(A.SEMICOLON_TOKEN = 27)] = "SEMICOLON_TOKEN"),
            (A[(A.LEFT_SQUARE_BRACKET_TOKEN = 28)] = "LEFT_SQUARE_BRACKET_TOKEN"),
            (A[(A.RIGHT_SQUARE_BRACKET_TOKEN = 29)] = "RIGHT_SQUARE_BRACKET_TOKEN"),
            (A[(A.UNICODE_RANGE_TOKEN = 30)] = "UNICODE_RANGE_TOKEN"),
            (A[(A.WHITESPACE_TOKEN = 31)] = "WHITESPACE_TOKEN"),
            (A[(A.EOF_TOKEN = 32)] = "EOF_TOKEN");
    })(U || (U = {}));
    var nA = function (A) {
            return A >= 48 && A <= 57;
        },
        sA = function (A) {
            return nA(A) || (A >= 65 && A <= 70) || (A >= 97 && A <= 102);
        },
        iA = function (A) {
            return 10 === A || 9 === A || 32 === A;
        },
        oA = function (A) {
            return (
                (function (A) {
                    return (
                        (function (A) {
                            return A >= 97 && A <= 122;
                        })(A) ||
                        (function (A) {
                            return A >= 65 && A <= 90;
                        })(A)
                    );
                })(A) ||
                (function (A) {
                    return A >= 128;
                })(A) ||
                95 === A
            );
        },
        BA = function (A) {
            return oA(A) || nA(A) || 45 === A;
        },
        aA = function (A) {
            return (A >= 0 && A <= 8) || 11 === A || (A >= 14 && A <= 31) || 127 === A;
        },
        cA = function (A, e) {
            return 92 === A && 10 !== e;
        },
        lA = function (A, e, t) {
            return 45 === A ? oA(e) || cA(e, t) : !!oA(A) || !(92 !== A || !cA(A, e));
        },
        QA = function (A, e, t) {
            return 43 === A || 45 === A ? !!nA(e) || (46 === e && nA(t)) : nA(46 === A ? e : A);
        },
        uA = function (A) {
            var e = 0,
                t = 1;
            (43 !== A[e] && 45 !== A[e]) || (45 === A[e] && (t = -1), e++);
            for (var r = []; nA(A[e]); ) r.push(A[e++]);
            var n = r.length ? parseInt(u.apply(void 0, r), 10) : 0;
            46 === A[e] && e++;
            for (var s = []; nA(A[e]); ) s.push(A[e++]);
            var i = s.length,
                o = i ? parseInt(u.apply(void 0, s), 10) : 0;
            (69 !== A[e] && 101 !== A[e]) || e++;
            var B = 1;
            (43 !== A[e] && 45 !== A[e]) || (45 === A[e] && (B = -1), e++);
            for (var a = []; nA(A[e]); ) a.push(A[e++]);
            var c = a.length ? parseInt(u.apply(void 0, a), 10) : 0;
            return t * (n + o * Math.pow(10, -i)) * Math.pow(10, B * c);
        },
        wA = { type: U.LEFT_PARENTHESIS_TOKEN },
        hA = { type: U.RIGHT_PARENTHESIS_TOKEN },
        gA = { type: U.COMMA_TOKEN },
        UA = { type: U.SUFFIX_MATCH_TOKEN },
        CA = { type: U.PREFIX_MATCH_TOKEN },
        dA = { type: U.COLUMN_TOKEN },
        EA = { type: U.DASH_MATCH_TOKEN },
        FA = { type: U.INCLUDE_MATCH_TOKEN },
        fA = { type: U.LEFT_CURLY_BRACKET_TOKEN },
        HA = { type: U.RIGHT_CURLY_BRACKET_TOKEN },
        pA = { type: U.SUBSTRING_MATCH_TOKEN },
        NA = { type: U.BAD_URL_TOKEN },
        KA = { type: U.BAD_STRING_TOKEN },
        mA = { type: U.CDO_TOKEN },
        IA = { type: U.CDC_TOKEN },
        TA = { type: U.COLON_TOKEN },
        vA = { type: U.SEMICOLON_TOKEN },
        LA = { type: U.LEFT_SQUARE_BRACKET_TOKEN },
        RA = { type: U.RIGHT_SQUARE_BRACKET_TOKEN },
        yA = { type: U.WHITESPACE_TOKEN },
        bA = { type: U.EOF_TOKEN },
        OA = (function () {
            function A() {
                this._value = [];
            }
            return (
                (A.prototype.write = function (A) {
                    this._value = this._value.concat(Q(A));
                }),
                (A.prototype.read = function () {
                    for (var A = [], e = this.consumeToken(); e !== bA; ) A.push(e), (e = this.consumeToken());
                    return A;
                }),
                (A.prototype.consumeToken = function () {
                    var A = this.consumeCodePoint();
                    switch (A) {
                        case 34:
                            return this.consumeStringToken(34);
                        case 35:
                            var e = this.peekCodePoint(0),
                                t = this.peekCodePoint(1),
                                r = this.peekCodePoint(2);
                            if (BA(e) || cA(t, r)) {
                                var n = lA(e, t, r) ? 2 : 1,
                                    s = this.consumeName();
                                return { type: U.HASH_TOKEN, value: s, flags: n };
                            }
                            break;
                        case 36:
                            if (61 === this.peekCodePoint(0)) return this.consumeCodePoint(), UA;
                            break;
                        case 39:
                            return this.consumeStringToken(39);
                        case 40:
                            return wA;
                        case 41:
                            return hA;
                        case 42:
                            if (61 === this.peekCodePoint(0)) return this.consumeCodePoint(), pA;
                            break;
                        case 43:
                            if (QA(A, this.peekCodePoint(0), this.peekCodePoint(1))) return this.reconsumeCodePoint(A), this.consumeNumericToken();
                            break;
                        case 44:
                            return gA;
                        case 45:
                            var i = A,
                                o = this.peekCodePoint(0),
                                B = this.peekCodePoint(1);
                            if (QA(i, o, B)) return this.reconsumeCodePoint(A), this.consumeNumericToken();
                            if (lA(i, o, B)) return this.reconsumeCodePoint(A), this.consumeIdentLikeToken();
                            if (45 === o && 62 === B) return this.consumeCodePoint(), this.consumeCodePoint(), IA;
                            break;
                        case 46:
                            if (QA(A, this.peekCodePoint(0), this.peekCodePoint(1))) return this.reconsumeCodePoint(A), this.consumeNumericToken();
                            break;
                        case 47:
                            if (42 === this.peekCodePoint(0))
                                for (this.consumeCodePoint(); ; ) {
                                    var a = this.consumeCodePoint();
                                    if (42 === a && 47 === (a = this.consumeCodePoint())) return this.consumeToken();
                                    if (-1 === a) return this.consumeToken();
                                }
                            break;
                        case 58:
                            return TA;
                        case 59:
                            return vA;
                        case 60:
                            if (33 === this.peekCodePoint(0) && 45 === this.peekCodePoint(1) && 45 === this.peekCodePoint(2)) return this.consumeCodePoint(), this.consumeCodePoint(), mA;
                            break;
                        case 64:
                            var c = this.peekCodePoint(0),
                                l = this.peekCodePoint(1),
                                Q = this.peekCodePoint(2);
                            if (lA(c, l, Q)) {
                                s = this.consumeName();
                                return { type: U.AT_KEYWORD_TOKEN, value: s };
                            }
                            break;
                        case 91:
                            return LA;
                        case 92:
                            if (cA(A, this.peekCodePoint(0))) return this.reconsumeCodePoint(A), this.consumeIdentLikeToken();
                            break;
                        case 93:
                            return RA;
                        case 61:
                            if (61 === this.peekCodePoint(0)) return this.consumeCodePoint(), CA;
                            break;
                        case 123:
                            return fA;
                        case 125:
                            return HA;
                        case 117:
                        case 85:
                            var w = this.peekCodePoint(0),
                                h = this.peekCodePoint(1);
                            return 43 !== w || (!sA(h) && 63 !== h) || (this.consumeCodePoint(), this.consumeUnicodeRangeToken()), this.reconsumeCodePoint(A), this.consumeIdentLikeToken();
                        case 124:
                            if (61 === this.peekCodePoint(0)) return this.consumeCodePoint(), EA;
                            if (124 === this.peekCodePoint(0)) return this.consumeCodePoint(), dA;
                            break;
                        case 126:
                            if (61 === this.peekCodePoint(0)) return this.consumeCodePoint(), FA;
                            break;
                        case -1:
                            return bA;
                    }
                    return iA(A)
                        ? (this.consumeWhiteSpace(), yA)
                        : nA(A)
                        ? (this.reconsumeCodePoint(A), this.consumeNumericToken())
                        : oA(A)
                        ? (this.reconsumeCodePoint(A), this.consumeIdentLikeToken())
                        : { type: U.DELIM_TOKEN, value: u(A) };
                }),
                (A.prototype.consumeCodePoint = function () {
                    var A = this._value.shift();
                    return void 0 === A ? -1 : A;
                }),
                (A.prototype.reconsumeCodePoint = function (A) {
                    this._value.unshift(A);
                }),
                (A.prototype.peekCodePoint = function (A) {
                    return A >= this._value.length ? -1 : this._value[A];
                }),
                (A.prototype.consumeUnicodeRangeToken = function () {
                    for (var A = [], e = this.consumeCodePoint(); sA(e) && A.length < 6; ) A.push(e), (e = this.consumeCodePoint());
                    for (var t = !1; 63 === e && A.length < 6; ) A.push(e), (e = this.consumeCodePoint()), (t = !0);
                    if (t) {
                        var r = parseInt(
                                u.apply(
                                    void 0,
                                    A.map(function (A) {
                                        return 63 === A ? 48 : A;
                                    })
                                ),
                                16
                            ),
                            n = parseInt(
                                u.apply(
                                    void 0,
                                    A.map(function (A) {
                                        return 63 === A ? 70 : A;
                                    })
                                ),
                                16
                            );
                        return { type: U.UNICODE_RANGE_TOKEN, start: r, end: n };
                    }
                    var s = parseInt(u.apply(void 0, A), 16);
                    if (45 === this.peekCodePoint(0) && sA(this.peekCodePoint(1))) {
                        this.consumeCodePoint(), (e = this.consumeCodePoint());
                        for (var i = []; sA(e) && i.length < 6; ) i.push(e), (e = this.consumeCodePoint());
                        n = parseInt(u.apply(void 0, i), 16);
                        return { type: U.UNICODE_RANGE_TOKEN, start: s, end: n };
                    }
                    return { type: U.UNICODE_RANGE_TOKEN, start: s, end: s };
                }),
                (A.prototype.consumeIdentLikeToken = function () {
                    var A = this.consumeName();
                    return "url" === A.toLowerCase() && 40 === this.peekCodePoint(0)
                        ? (this.consumeCodePoint(), this.consumeUrlToken())
                        : 40 === this.peekCodePoint(0)
                        ? (this.consumeCodePoint(), { type: U.FUNCTION_TOKEN, value: A })
                        : { type: U.IDENT_TOKEN, value: A };
                }),
                (A.prototype.consumeUrlToken = function () {
                    var A = [];
                    if ((this.consumeWhiteSpace(), -1 === this.peekCodePoint(0))) return { type: U.URL_TOKEN, value: "" };
                    var e = this.peekCodePoint(0);
                    if (39 === e || 34 === e) {
                        var t = this.consumeStringToken(this.consumeCodePoint());
                        return t.type === U.STRING_TOKEN && (this.consumeWhiteSpace(), -1 === this.peekCodePoint(0) || 41 === this.peekCodePoint(0))
                            ? (this.consumeCodePoint(), { type: U.URL_TOKEN, value: t.value })
                            : (this.consumeBadUrlRemnants(), NA);
                    }
                    for (;;) {
                        var r = this.consumeCodePoint();
                        if (-1 === r || 41 === r) return { type: U.URL_TOKEN, value: u.apply(void 0, A) };
                        if (iA(r))
                            return this.consumeWhiteSpace(), -1 === this.peekCodePoint(0) || 41 === this.peekCodePoint(0) ? (this.consumeCodePoint(), { type: U.URL_TOKEN, value: u.apply(void 0, A) }) : (this.consumeBadUrlRemnants(), NA);
                        if (34 === r || 39 === r || 40 === r || aA(r)) return this.consumeBadUrlRemnants(), NA;
                        if (92 === r) {
                            if (!cA(r, this.peekCodePoint(0))) return this.consumeBadUrlRemnants(), NA;
                            A.push(this.consumeEscapedCodePoint());
                        } else A.push(r);
                    }
                }),
                (A.prototype.consumeWhiteSpace = function () {
                    for (; iA(this.peekCodePoint(0)); ) this.consumeCodePoint();
                }),
                (A.prototype.consumeBadUrlRemnants = function () {
                    for (;;) {
                        var A = this.consumeCodePoint();
                        if (41 === A || -1 === A) return;
                        cA(A, this.peekCodePoint(0)) && this.consumeEscapedCodePoint();
                    }
                }),
                (A.prototype.consumeStringSlice = function (A) {
                    for (var e = ""; A > 0; ) {
                        var t = Math.min(6e4, A);
                        (e += u.apply(void 0, this._value.splice(0, t))), (A -= t);
                    }
                    return this._value.shift(), e;
                }),
                (A.prototype.consumeStringToken = function (A) {
                    for (var e = "", t = 0; ; ) {
                        var r = this._value[t];
                        if (-1 === r || void 0 === r || r === A) return (e += this.consumeStringSlice(t)), { type: U.STRING_TOKEN, value: e };
                        if (10 === r) return this._value.splice(0, t), KA;
                        if (92 === r) {
                            var n = this._value[t + 1];
                            -1 !== n && void 0 !== n && (10 === n ? ((e += this.consumeStringSlice(t)), (t = -1), this._value.shift()) : cA(r, n) && ((e += this.consumeStringSlice(t)), (e += u(this.consumeEscapedCodePoint())), (t = -1)));
                        }
                        t++;
                    }
                }),
                (A.prototype.consumeNumber = function () {
                    var A = [],
                        e = 4,
                        t = this.peekCodePoint(0);
                    for ((43 !== t && 45 !== t) || A.push(this.consumeCodePoint()); nA(this.peekCodePoint(0)); ) A.push(this.consumeCodePoint());
                    t = this.peekCodePoint(0);
                    var r = this.peekCodePoint(1);
                    if (46 === t && nA(r)) for (A.push(this.consumeCodePoint(), this.consumeCodePoint()), e = 8; nA(this.peekCodePoint(0)); ) A.push(this.consumeCodePoint());
                    (t = this.peekCodePoint(0)), (r = this.peekCodePoint(1));
                    var n = this.peekCodePoint(2);
                    if ((69 === t || 101 === t) && (((43 === r || 45 === r) && nA(n)) || nA(r))) for (A.push(this.consumeCodePoint(), this.consumeCodePoint()), e = 8; nA(this.peekCodePoint(0)); ) A.push(this.consumeCodePoint());
                    return [uA(A), e];
                }),
                (A.prototype.consumeNumericToken = function () {
                    var A = this.consumeNumber(),
                        e = A[0],
                        t = A[1],
                        r = this.peekCodePoint(0),
                        n = this.peekCodePoint(1),
                        s = this.peekCodePoint(2);
                    if (lA(r, n, s)) {
                        var i = this.consumeName();
                        return { type: U.DIMENSION_TOKEN, number: e, flags: t, unit: i };
                    }
                    return 37 === r ? (this.consumeCodePoint(), { type: U.PERCENTAGE_TOKEN, number: e, flags: t }) : { type: U.NUMBER_TOKEN, number: e, flags: t };
                }),
                (A.prototype.consumeEscapedCodePoint = function () {
                    var A = this.consumeCodePoint();
                    if (sA(A)) {
                        for (var e = u(A); sA(this.peekCodePoint(0)) && e.length < 6; ) e += u(this.consumeCodePoint());
                        iA(this.peekCodePoint(0)) && this.consumeCodePoint();
                        var t = parseInt(e, 16);
                        return 0 === t ||
                            (function (A) {
                                return A >= 55296 && A <= 57343;
                            })(t) ||
                            t > 1114111
                            ? 65533
                            : t;
                    }
                    return -1 === A ? 65533 : A;
                }),
                (A.prototype.consumeName = function () {
                    for (var A = ""; ; ) {
                        var e = this.consumeCodePoint();
                        if (BA(e)) A += u(e);
                        else {
                            if (!cA(e, this.peekCodePoint(0))) return this.reconsumeCodePoint(e), A;
                            A += u(this.consumeEscapedCodePoint());
                        }
                    }
                }),
                A
            );
        })(),
        SA = (function () {
            function A(A) {
                this._tokens = A;
            }
            return (
                (A.create = function (e) {
                    var t = new OA();
                    return t.write(e), new A(t.read());
                }),
                (A.parseValue = function (e) {
                    return A.create(e).parseComponentValue();
                }),
                (A.parseValues = function (e) {
                    return A.create(e).parseComponentValues();
                }),
                (A.prototype.parseComponentValue = function () {
                    for (var A = this.consumeToken(); A.type === U.WHITESPACE_TOKEN; ) A = this.consumeToken();
                    if (A.type === U.EOF_TOKEN) throw new SyntaxError("Error parsing CSS component value, unexpected EOF");
                    this.reconsumeToken(A);
                    var e = this.consumeComponentValue();
                    do {
                        A = this.consumeToken();
                    } while (A.type === U.WHITESPACE_TOKEN);
                    if (A.type === U.EOF_TOKEN) return e;
                    throw new SyntaxError("Error parsing CSS component value, multiple values found when expecting only one");
                }),
                (A.prototype.parseComponentValues = function () {
                    for (var A = []; ; ) {
                        var e = this.consumeComponentValue();
                        if (e.type === U.EOF_TOKEN) return A;
                        A.push(e), A.push();
                    }
                }),
                (A.prototype.consumeComponentValue = function () {
                    var A = this.consumeToken();
                    switch (A.type) {
                        case U.LEFT_CURLY_BRACKET_TOKEN:
                        case U.LEFT_SQUARE_BRACKET_TOKEN:
                        case U.LEFT_PARENTHESIS_TOKEN:
                            return this.consumeSimpleBlock(A.type);
                        case U.FUNCTION_TOKEN:
                            return this.consumeFunction(A);
                    }
                    return A;
                }),
                (A.prototype.consumeSimpleBlock = function (A) {
                    for (var e = { type: A, values: [] }, t = this.consumeToken(); ; ) {
                        if (t.type === U.EOF_TOKEN || JA(t, A)) return e;
                        this.reconsumeToken(t), e.values.push(this.consumeComponentValue()), (t = this.consumeToken());
                    }
                }),
                (A.prototype.consumeFunction = function (A) {
                    for (var e = { name: A.value, values: [], type: U.FUNCTION }; ; ) {
                        var t = this.consumeToken();
                        if (t.type === U.EOF_TOKEN || t.type === U.RIGHT_PARENTHESIS_TOKEN) return e;
                        this.reconsumeToken(t), e.values.push(this.consumeComponentValue());
                    }
                }),
                (A.prototype.consumeToken = function () {
                    var A = this._tokens.shift();
                    return void 0 === A ? bA : A;
                }),
                (A.prototype.reconsumeToken = function (A) {
                    this._tokens.unshift(A);
                }),
                A
            );
        })(),
        MA = function (A) {
            return A.type === U.DIMENSION_TOKEN;
        },
        DA = function (A) {
            return A.type === U.NUMBER_TOKEN;
        },
        _A = function (A) {
            return A.type === U.IDENT_TOKEN;
        },
        xA = function (A) {
            return A.type === U.STRING_TOKEN;
        },
        PA = function (A, e) {
            return _A(A) && A.value === e;
        },
        VA = function (A) {
            return A.type !== U.WHITESPACE_TOKEN;
        },
        zA = function (A) {
            return A.type !== U.WHITESPACE_TOKEN && A.type !== U.COMMA_TOKEN;
        },
        XA = function (A) {
            var e = [],
                t = [];
            return (
                A.forEach(function (A) {
                    if (A.type === U.COMMA_TOKEN) {
                        if (0 === t.length) throw new Error("Error parsing function args, zero tokens for arg");
                        return e.push(t), void (t = []);
                    }
                    A.type !== U.WHITESPACE_TOKEN && t.push(A);
                }),
                t.length && e.push(t),
                e
            );
        },
        JA = function (A, e) {
            return (
                (e === U.LEFT_CURLY_BRACKET_TOKEN && A.type === U.RIGHT_CURLY_BRACKET_TOKEN) ||
                (e === U.LEFT_SQUARE_BRACKET_TOKEN && A.type === U.RIGHT_SQUARE_BRACKET_TOKEN) ||
                (e === U.LEFT_PARENTHESIS_TOKEN && A.type === U.RIGHT_PARENTHESIS_TOKEN)
            );
        },
        GA = function (A) {
            return A.type === U.NUMBER_TOKEN || A.type === U.DIMENSION_TOKEN;
        },
        kA = function (A) {
            return A.type === U.PERCENTAGE_TOKEN || GA(A);
        },
        WA = function (A) {
            return A.length > 1 ? [A[0], A[1]] : [A[0]];
        },
        YA = { type: U.NUMBER_TOKEN, number: 0, flags: 4 },
        qA = { type: U.PERCENTAGE_TOKEN, number: 50, flags: 4 },
        jA = { type: U.PERCENTAGE_TOKEN, number: 100, flags: 4 },
        ZA = function (A, e, t) {
            var r = A[0],
                n = A[1];
            return [$A(r, e), $A(void 0 !== n ? n : r, t)];
        },
        $A = function (A, e) {
            if (A.type === U.PERCENTAGE_TOKEN) return (A.number / 100) * e;
            if (MA(A))
                switch (A.unit) {
                    case "rem":
                    case "em":
                        return 16 * A.number;
                    case "px":
                    default:
                        return A.number;
                }
            return A.number;
        },
        Ae = function (A) {
            if (A.type === U.DIMENSION_TOKEN)
                switch (A.unit) {
                    case "deg":
                        return (Math.PI * A.number) / 180;
                    case "grad":
                        return (Math.PI / 200) * A.number;
                    case "rad":
                        return A.number;
                    case "turn":
                        return 2 * Math.PI * A.number;
                }
            throw new Error("Unsupported angle type");
        },
        ee = function (A) {
            return A.type === U.DIMENSION_TOKEN && ("deg" === A.unit || "grad" === A.unit || "rad" === A.unit || "turn" === A.unit);
        },
        te = function (A) {
            switch (
                A.filter(_A)
                    .map(function (A) {
                        return A.value;
                    })
                    .join(" ")
            ) {
                case "to bottom right":
                case "to right bottom":
                case "left top":
                case "top left":
                    return [YA, YA];
                case "to top":
                case "bottom":
                    return re(0);
                case "to bottom left":
                case "to left bottom":
                case "right top":
                case "top right":
                    return [YA, jA];
                case "to right":
                case "left":
                    return re(90);
                case "to top left":
                case "to left top":
                case "right bottom":
                case "bottom right":
                    return [jA, jA];
                case "to bottom":
                case "top":
                    return re(180);
                case "to top right":
                case "to right top":
                case "left bottom":
                case "bottom left":
                    return [jA, YA];
                case "to left":
                case "right":
                    return re(270);
            }
            return 0;
        },
        re = function (A) {
            return (Math.PI * A) / 180;
        },
        ne = function (A) {
            if (A.type === U.FUNCTION) {
                var e = we[A.name];
                if (void 0 === e) throw new Error('Attempting to parse an unsupported color function "' + A.name + '"');
                return e(A.values);
            }
            if (A.type === U.HASH_TOKEN) {
                if (3 === A.value.length) {
                    var t = A.value.substring(0, 1),
                        r = A.value.substring(1, 2),
                        n = A.value.substring(2, 3);
                    return oe(parseInt(t + t, 16), parseInt(r + r, 16), parseInt(n + n, 16), 1);
                }
                if (4 === A.value.length) {
                    (t = A.value.substring(0, 1)), (r = A.value.substring(1, 2)), (n = A.value.substring(2, 3));
                    var s = A.value.substring(3, 4);
                    return oe(parseInt(t + t, 16), parseInt(r + r, 16), parseInt(n + n, 16), parseInt(s + s, 16) / 255);
                }
                if (6 === A.value.length) {
                    (t = A.value.substring(0, 2)), (r = A.value.substring(2, 4)), (n = A.value.substring(4, 6));
                    return oe(parseInt(t, 16), parseInt(r, 16), parseInt(n, 16), 1);
                }
                if (8 === A.value.length) {
                    (t = A.value.substring(0, 2)), (r = A.value.substring(2, 4)), (n = A.value.substring(4, 6)), (s = A.value.substring(6, 8));
                    return oe(parseInt(t, 16), parseInt(r, 16), parseInt(n, 16), parseInt(s, 16) / 255);
                }
            }
            if (A.type === U.IDENT_TOKEN) {
                var i = he[A.value.toUpperCase()];
                if (void 0 !== i) return i;
            }
            return he.TRANSPARENT;
        },
        se = function (A) {
            return 0 == (255 & A);
        },
        ie = function (A) {
            var e = 255 & A,
                t = 255 & (A >> 8),
                r = 255 & (A >> 16),
                n = 255 & (A >> 24);
            return e < 255 ? "rgba(" + n + "," + r + "," + t + "," + e / 255 + ")" : "rgb(" + n + "," + r + "," + t + ")";
        },
        oe = function (A, e, t, r) {
            return ((A << 24) | (e << 16) | (t << 8) | (Math.round(255 * r) << 0)) >>> 0;
        },
        Be = function (A, e) {
            if (A.type === U.NUMBER_TOKEN) return A.number;
            if (A.type === U.PERCENTAGE_TOKEN) {
                var t = 3 === e ? 1 : 255;
                return 3 === e ? (A.number / 100) * t : Math.round((A.number / 100) * t);
            }
            return 0;
        },
        ae = function (A) {
            var e = A.filter(zA);
            if (3 === e.length) {
                var t = e.map(Be),
                    r = t[0],
                    n = t[1],
                    s = t[2];
                return oe(r, n, s, 1);
            }
            if (4 === e.length) {
                var i = e.map(Be),
                    o = ((r = i[0]), (n = i[1]), (s = i[2]), i[3]);
                return oe(r, n, s, o);
            }
            return 0;
        };
    function ce(A, e, t) {
        return t < 0 && (t += 1), t >= 1 && (t -= 1), t < 1 / 6 ? (e - A) * t * 6 + A : t < 0.5 ? e : t < 2 / 3 ? 6 * (e - A) * (2 / 3 - t) + A : A;
    }
    var le,
        Qe,
        ue = function (A) {
            var e = A.filter(zA),
                t = e[0],
                r = e[1],
                n = e[2],
                s = e[3],
                i = (t.type === U.NUMBER_TOKEN ? re(t.number) : Ae(t)) / (2 * Math.PI),
                o = kA(r) ? r.number / 100 : 0,
                B = kA(n) ? n.number / 100 : 0,
                a = void 0 !== s && kA(s) ? $A(s, 1) : 1;
            if (0 === o) return oe(255 * B, 255 * B, 255 * B, 1);
            var c = B <= 0.5 ? B * (o + 1) : B + o - B * o,
                l = 2 * B - c,
                Q = ce(l, c, i + 1 / 3),
                u = ce(l, c, i),
                w = ce(l, c, i - 1 / 3);
            return oe(255 * Q, 255 * u, 255 * w, a);
        },
        we = { hsl: ue, hsla: ue, rgb: ae, rgba: ae },
        he = {
            ALICEBLUE: 4042850303,
            ANTIQUEWHITE: 4209760255,
            AQUA: 16777215,
            AQUAMARINE: 2147472639,
            AZURE: 4043309055,
            BEIGE: 4126530815,
            BISQUE: 4293182719,
            BLACK: 255,
            BLANCHEDALMOND: 4293643775,
            BLUE: 65535,
            BLUEVIOLET: 2318131967,
            BROWN: 2771004159,
            BURLYWOOD: 3736635391,
            CADETBLUE: 1604231423,
            CHARTREUSE: 2147418367,
            CHOCOLATE: 3530104575,
            CORAL: 4286533887,
            CORNFLOWERBLUE: 1687547391,
            CORNSILK: 4294499583,
            CRIMSON: 3692313855,
            CYAN: 16777215,
            DARKBLUE: 35839,
            DARKCYAN: 9145343,
            DARKGOLDENROD: 3095837695,
            DARKGRAY: 2846468607,
            DARKGREEN: 6553855,
            DARKGREY: 2846468607,
            DARKKHAKI: 3182914559,
            DARKMAGENTA: 2332068863,
            DARKOLIVEGREEN: 1433087999,
            DARKORANGE: 4287365375,
            DARKORCHID: 2570243327,
            DARKRED: 2332033279,
            DARKSALMON: 3918953215,
            DARKSEAGREEN: 2411499519,
            DARKSLATEBLUE: 1211993087,
            DARKSLATEGRAY: 793726975,
            DARKSLATEGREY: 793726975,
            DARKTURQUOISE: 13554175,
            DARKVIOLET: 2483082239,
            DEEPPINK: 4279538687,
            DEEPSKYBLUE: 12582911,
            DIMGRAY: 1768516095,
            DIMGREY: 1768516095,
            DODGERBLUE: 512819199,
            FIREBRICK: 2988581631,
            FLORALWHITE: 4294635775,
            FORESTGREEN: 579543807,
            FUCHSIA: 4278255615,
            GAINSBORO: 3705462015,
            GHOSTWHITE: 4177068031,
            GOLD: 4292280575,
            GOLDENROD: 3668254975,
            GRAY: 2155905279,
            GREEN: 8388863,
            GREENYELLOW: 2919182335,
            GREY: 2155905279,
            HONEYDEW: 4043305215,
            HOTPINK: 4285117695,
            INDIANRED: 3445382399,
            INDIGO: 1258324735,
            IVORY: 4294963455,
            KHAKI: 4041641215,
            LAVENDER: 3873897215,
            LAVENDERBLUSH: 4293981695,
            LAWNGREEN: 2096890111,
            LEMONCHIFFON: 4294626815,
            LIGHTBLUE: 2916673279,
            LIGHTCORAL: 4034953471,
            LIGHTCYAN: 3774873599,
            LIGHTGOLDENRODYELLOW: 4210742015,
            LIGHTGRAY: 3553874943,
            LIGHTGREEN: 2431553791,
            LIGHTGREY: 3553874943,
            LIGHTPINK: 4290167295,
            LIGHTSALMON: 4288707327,
            LIGHTSEAGREEN: 548580095,
            LIGHTSKYBLUE: 2278488831,
            LIGHTSLATEGRAY: 2005441023,
            LIGHTSLATEGREY: 2005441023,
            LIGHTSTEELBLUE: 2965692159,
            LIGHTYELLOW: 4294959359,
            LIME: 16711935,
            LIMEGREEN: 852308735,
            LINEN: 4210091775,
            MAGENTA: 4278255615,
            MAROON: 2147483903,
            MEDIUMAQUAMARINE: 1724754687,
            MEDIUMBLUE: 52735,
            MEDIUMORCHID: 3126187007,
            MEDIUMPURPLE: 2473647103,
            MEDIUMSEAGREEN: 1018393087,
            MEDIUMSLATEBLUE: 2070474495,
            MEDIUMSPRINGGREEN: 16423679,
            MEDIUMTURQUOISE: 1221709055,
            MEDIUMVIOLETRED: 3340076543,
            MIDNIGHTBLUE: 421097727,
            MINTCREAM: 4127193855,
            MISTYROSE: 4293190143,
            MOCCASIN: 4293178879,
            NAVAJOWHITE: 4292783615,
            NAVY: 33023,
            OLDLACE: 4260751103,
            OLIVE: 2155872511,
            OLIVEDRAB: 1804477439,
            ORANGE: 4289003775,
            ORANGERED: 4282712319,
            ORCHID: 3664828159,
            PALEGOLDENROD: 4008225535,
            PALEGREEN: 2566625535,
            PALETURQUOISE: 2951671551,
            PALEVIOLETRED: 3681588223,
            PAPAYAWHIP: 4293907967,
            PEACHPUFF: 4292524543,
            PERU: 3448061951,
            PINK: 4290825215,
            PLUM: 3718307327,
            POWDERBLUE: 2967529215,
            PURPLE: 2147516671,
            REBECCAPURPLE: 1714657791,
            RED: 4278190335,
            ROSYBROWN: 3163525119,
            ROYALBLUE: 1097458175,
            SADDLEBROWN: 2336560127,
            SALMON: 4202722047,
            SANDYBROWN: 4104413439,
            SEAGREEN: 780883967,
            SEASHELL: 4294307583,
            SIENNA: 2689740287,
            SILVER: 3233857791,
            SKYBLUE: 2278484991,
            SLATEBLUE: 1784335871,
            SLATEGRAY: 1887473919,
            SLATEGREY: 1887473919,
            SNOW: 4294638335,
            SPRINGGREEN: 16744447,
            STEELBLUE: 1182971135,
            TAN: 3535047935,
            TEAL: 8421631,
            THISTLE: 3636451583,
            TOMATO: 4284696575,
            TRANSPARENT: 0,
            TURQUOISE: 1088475391,
            VIOLET: 4001558271,
            WHEAT: 4125012991,
            WHITE: 4294967295,
            WHITESMOKE: 4126537215,
            YELLOW: 4294902015,
            YELLOWGREEN: 2597139199,
        };
    !(function (A) {
        (A[(A.VALUE = 0)] = "VALUE"), (A[(A.LIST = 1)] = "LIST"), (A[(A.IDENT_VALUE = 2)] = "IDENT_VALUE"), (A[(A.TYPE_VALUE = 3)] = "TYPE_VALUE"), (A[(A.TOKEN_VALUE = 4)] = "TOKEN_VALUE");
    })(le || (le = {})),
        (function (A) {
            (A[(A.BORDER_BOX = 0)] = "BORDER_BOX"), (A[(A.PADDING_BOX = 1)] = "PADDING_BOX"), (A[(A.CONTENT_BOX = 2)] = "CONTENT_BOX");
        })(Qe || (Qe = {}));
    var ge,
        Ue = {
            name: "background-clip",
            initialValue: "border-box",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return A.map(function (A) {
                    if (_A(A))
                        switch (A.value) {
                            case "padding-box":
                                return Qe.PADDING_BOX;
                            case "content-box":
                                return Qe.CONTENT_BOX;
                        }
                    return Qe.BORDER_BOX;
                });
            },
        },
        Ce = { name: "background-color", initialValue: "transparent", prefix: !1, type: le.TYPE_VALUE, format: "color" },
        de = function (A) {
            var e = ne(A[0]),
                t = A[1];
            return t && kA(t) ? { color: e, stop: t } : { color: e, stop: null };
        },
        Ee = function (A, e) {
            var t = A[0],
                r = A[A.length - 1];
            null === t.stop && (t.stop = YA), null === r.stop && (r.stop = jA);
            for (var n = [], s = 0, i = 0; i < A.length; i++) {
                var o = A[i].stop;
                if (null !== o) {
                    var B = $A(o, e);
                    B > s ? n.push(B) : n.push(s), (s = B);
                } else n.push(null);
            }
            var a = null;
            for (i = 0; i < n.length; i++) {
                var c = n[i];
                if (null === c) null === a && (a = i);
                else if (null !== a) {
                    for (var l = i - a, Q = (c - n[a - 1]) / (l + 1), u = 1; u <= l; u++) n[a + u - 1] = Q * u;
                    a = null;
                }
            }
            return A.map(function (A, t) {
                return { color: A.color, stop: Math.max(Math.min(1, n[t] / e), 0) };
            });
        },
        Fe = function (A, e, t) {
            var r =
                    "number" == typeof A
                        ? A
                        : (function (A, e, t) {
                              var r = e / 2,
                                  n = t / 2,
                                  s = $A(A[0], e) - r,
                                  i = n - $A(A[1], t);
                              return (Math.atan2(i, s) + 2 * Math.PI) % (2 * Math.PI);
                          })(A, e, t),
                n = Math.abs(e * Math.sin(r)) + Math.abs(t * Math.cos(r)),
                s = e / 2,
                i = t / 2,
                o = n / 2,
                B = Math.sin(r - Math.PI / 2) * o,
                a = Math.cos(r - Math.PI / 2) * o;
            return [n, s - a, s + a, i - B, i + B];
        },
        fe = function (A, e) {
            return Math.sqrt(A * A + e * e);
        },
        He = function (A, e, t, r, n) {
            return [
                [0, 0],
                [0, e],
                [A, 0],
                [A, e],
            ].reduce(
                function (A, e) {
                    var s = e[0],
                        i = e[1],
                        o = fe(t - s, r - i);
                    return (n ? o < A.optimumDistance : o > A.optimumDistance) ? { optimumCorner: e, optimumDistance: o } : A;
                },
                { optimumDistance: n ? 1 / 0 : -1 / 0, optimumCorner: null }
            ).optimumCorner;
        },
        pe = function (A) {
            var e = re(180),
                t = [];
            return (
                XA(A).forEach(function (A, r) {
                    if (0 === r) {
                        var n = A[0];
                        if (n.type === U.IDENT_TOKEN && -1 !== ["top", "left", "right", "bottom"].indexOf(n.value)) return void (e = te(A));
                        if (ee(n)) return void (e = (Ae(n) + re(270)) % re(360));
                    }
                    var s = de(A);
                    t.push(s);
                }),
                { angle: e, stops: t, type: ge.LINEAR_GRADIENT }
            );
        },
        Ne = function (A) {
            return 0 === A[0] && 255 === A[1] && 0 === A[2] && 255 === A[3];
        },
        Ke = function (A, e, t, r, n) {
            var s = "http://www.w3.org/2000/svg",
                i = document.createElementNS(s, "svg"),
                o = document.createElementNS(s, "foreignObject");
            return (
                i.setAttributeNS(null, "width", A.toString()),
                i.setAttributeNS(null, "height", e.toString()),
                o.setAttributeNS(null, "width", "100%"),
                o.setAttributeNS(null, "height", "100%"),
                o.setAttributeNS(null, "x", t.toString()),
                o.setAttributeNS(null, "y", r.toString()),
                o.setAttributeNS(null, "externalResourcesRequired", "true"),
                i.appendChild(o),
                o.appendChild(n),
                i
            );
        },
        me = function (A) {
            return new Promise(function (e, t) {
                var r = new Image();
                (r.onload = function () {
                    return e(r);
                }),
                    (r.onerror = t),
                    (r.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(A)));
            });
        },
        Ie = {
            get SUPPORT_RANGE_BOUNDS() {
                var A = (function (A) {
                    if (A.createRange) {
                        var e = A.createRange();
                        if (e.getBoundingClientRect) {
                            var t = A.createElement("boundtest");
                            (t.style.height = "123px"), (t.style.display = "block"), A.body.appendChild(t), e.selectNode(t);
                            var r = e.getBoundingClientRect(),
                                n = Math.round(r.height);
                            if ((A.body.removeChild(t), 123 === n)) return !0;
                        }
                    }
                    return !1;
                })(document);
                return Object.defineProperty(Ie, "SUPPORT_RANGE_BOUNDS", { value: A }), A;
            },
            get SUPPORT_SVG_DRAWING() {
                var A = (function (A) {
                    var e = new Image(),
                        t = A.createElement("canvas"),
                        r = t.getContext("2d");
                    if (!r) return !1;
                    e.src = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'></svg>";
                    try {
                        r.drawImage(e, 0, 0), t.toDataURL();
                    } catch (A) {
                        return !1;
                    }
                    return !0;
                })(document);
                return Object.defineProperty(Ie, "SUPPORT_SVG_DRAWING", { value: A }), A;
            },
            get SUPPORT_FOREIGNOBJECT_DRAWING() {
                var A =
                    "function" == typeof Array.from && "function" == typeof window.fetch
                        ? (function (A) {
                              var e = A.createElement("canvas");
                              (e.width = 100), (e.height = 100);
                              var t = e.getContext("2d");
                              if (!t) return Promise.reject(!1);
                              (t.fillStyle = "rgb(0, 255, 0)"), t.fillRect(0, 0, 100, 100);
                              var r = new Image(),
                                  n = e.toDataURL();
                              r.src = n;
                              var s = Ke(100, 100, 0, 0, r);
                              return (
                                  (t.fillStyle = "red"),
                                  t.fillRect(0, 0, 100, 100),
                                  me(s)
                                      .then(function (e) {
                                          t.drawImage(e, 0, 0);
                                          var r = t.getImageData(0, 0, 100, 100).data;
                                          (t.fillStyle = "red"), t.fillRect(0, 0, 100, 100);
                                          var s = A.createElement("div");
                                          return (s.style.backgroundImage = "url(" + n + ")"), (s.style.height = "100px"), Ne(r) ? me(Ke(100, 100, 0, 0, s)) : Promise.reject(!1);
                                      })
                                      .then(function (A) {
                                          return t.drawImage(A, 0, 0), Ne(t.getImageData(0, 0, 100, 100).data);
                                      })
                                      .catch(function () {
                                          return !1;
                                      })
                              );
                          })(document)
                        : Promise.resolve(!1);
                return Object.defineProperty(Ie, "SUPPORT_FOREIGNOBJECT_DRAWING", { value: A }), A;
            },
            get SUPPORT_CORS_IMAGES() {
                var A = void 0 !== new Image().crossOrigin;
                return Object.defineProperty(Ie, "SUPPORT_CORS_IMAGES", { value: A }), A;
            },
            get SUPPORT_RESPONSE_TYPE() {
                var A = "string" == typeof new XMLHttpRequest().responseType;
                return Object.defineProperty(Ie, "SUPPORT_RESPONSE_TYPE", { value: A }), A;
            },
            get SUPPORT_CORS_XHR() {
                var A = "withCredentials" in new XMLHttpRequest();
                return Object.defineProperty(Ie, "SUPPORT_CORS_XHR", { value: A }), A;
            },
        },
        Te = (function () {
            function A(A) {
                (this.id = A), (this.start = Date.now());
            }
            return (
                (A.prototype.debug = function () {
                    for (var A = [], e = 0; e < arguments.length; e++) A[e] = arguments[e];
                    "undefined" != typeof window && window.console && "function" == typeof console.debug ? console.debug.apply(console, [this.id, this.getTime() + "ms"].concat(A)) : this.info.apply(this, A);
                }),
                (A.prototype.getTime = function () {
                    return Date.now() - this.start;
                }),
                (A.create = function (e) {
                    A.instances[e] = new A(e);
                }),
                (A.destroy = function (e) {
                    delete A.instances[e];
                }),
                (A.getInstance = function (e) {
                    var t = A.instances[e];
                    if (void 0 === t) throw new Error("No logger instance found with id " + e);
                    return t;
                }),
                (A.prototype.info = function () {
                    for (var A = [], e = 0; e < arguments.length; e++) A[e] = arguments[e];
                    "undefined" != typeof window && window.console && "function" == typeof console.info && console.info.apply(console, [this.id, this.getTime() + "ms"].concat(A));
                }),
                (A.prototype.error = function () {
                    for (var A = [], e = 0; e < arguments.length; e++) A[e] = arguments[e];
                    "undefined" != typeof window && window.console && "function" == typeof console.error ? console.error.apply(console, [this.id, this.getTime() + "ms"].concat(A)) : this.info.apply(this, A);
                }),
                (A.instances = {}),
                A
            );
        })(),
        ve = (function () {
            function A() {}
            return (
                (A.create = function (e, t) {
                    return (A._caches[e] = new Le(e, t));
                }),
                (A.destroy = function (e) {
                    delete A._caches[e];
                }),
                (A.open = function (e) {
                    var t = A._caches[e];
                    if (void 0 !== t) return t;
                    throw new Error('Cache with key "' + e + '" not found');
                }),
                (A.getOrigin = function (e) {
                    var t = A._link;
                    return t ? ((t.href = e), (t.href = t.href), t.protocol + t.hostname + t.port) : "about:blank";
                }),
                (A.isSameOrigin = function (e) {
                    return A.getOrigin(e) === A._origin;
                }),
                (A.setContext = function (e) {
                    (A._link = e.document.createElement("a")), (A._origin = A.getOrigin(e.location.href));
                }),
                (A.getInstance = function () {
                    var e = A._current;
                    if (null === e) throw new Error("No cache instance attached");
                    return e;
                }),
                (A.attachInstance = function (e) {
                    A._current = e;
                }),
                (A.detachInstance = function () {
                    A._current = null;
                }),
                (A._caches = {}),
                (A._origin = "about:blank"),
                (A._current = null),
                A
            );
        })(),
        Le = (function () {
            function A(A, e) {
                (this.id = A), (this._options = e), (this._cache = {});
            }
            return (
                (A.prototype.addImage = function (A) {
                    var e = Promise.resolve();
                    return this.has(A) ? e : De(A) || Oe(A) ? ((this._cache[A] = this.loadImage(A)), e) : e;
                }),
                (A.prototype.match = function (A) {
                    return this._cache[A];
                }),
                (A.prototype.loadImage = function (A) {
                    return B(this, void 0, void 0, function () {
                        var e,
                            t,
                            r,
                            n,
                            s = this;
                        return a(this, function (i) {
                            switch (i.label) {
                                case 0:
                                    return (
                                        (e = ve.isSameOrigin(A)),
                                        (t = !Se(A) && !0 === this._options.useCORS && Ie.SUPPORT_CORS_IMAGES && !e),
                                        (r = !Se(A) && !e && "string" == typeof this._options.proxy && Ie.SUPPORT_CORS_XHR && !t),
                                        e || !1 !== this._options.allowTaint || Se(A) || r || t ? ((n = A), r ? [4, this.proxy(n)] : [3, 2]) : [2]
                                    );
                                case 1:
                                    (n = i.sent()), (i.label = 2);
                                case 2:
                                    return (
                                        Te.getInstance(this.id).debug("Added image " + A.substring(0, 256)),
                                        [
                                            4,
                                            new Promise(function (A, e) {
                                                var r = new Image();
                                                (r.onload = function () {
                                                    return A(r);
                                                }),
                                                    (r.onerror = e),
                                                    (Me(n) || t) && (r.crossOrigin = "anonymous"),
                                                    (r.src = n),
                                                    !0 === r.complete &&
                                                        setTimeout(function () {
                                                            return A(r);
                                                        }, 500),
                                                    s._options.imageTimeout > 0 &&
                                                        setTimeout(function () {
                                                            return e("Timed out (" + s._options.imageTimeout + "ms) loading image");
                                                        }, s._options.imageTimeout);
                                            }),
                                        ]
                                    );
                                case 3:
                                    return [2, i.sent()];
                            }
                        });
                    });
                }),
                (A.prototype.has = function (A) {
                    return void 0 !== this._cache[A];
                }),
                (A.prototype.keys = function () {
                    return Promise.resolve(Object.keys(this._cache));
                }),
                (A.prototype.proxy = function (A) {
                    var e = this,
                        t = this._options.proxy;
                    if (!t) throw new Error("No proxy defined");
                    var r = A.substring(0, 256);
                    return new Promise(function (n, s) {
                        var i = Ie.SUPPORT_RESPONSE_TYPE ? "blob" : "text",
                            o = new XMLHttpRequest();
                        if (
                            ((o.onload = function () {
                                if (200 === o.status)
                                    if ("text" === i) n(o.response);
                                    else {
                                        var A = new FileReader();
                                        A.addEventListener(
                                            "load",
                                            function () {
                                                return n(A.result);
                                            },
                                            !1
                                        ),
                                            A.addEventListener(
                                                "error",
                                                function (A) {
                                                    return s(A);
                                                },
                                                !1
                                            ),
                                            A.readAsDataURL(o.response);
                                    }
                                else s("Failed to proxy resource " + r + " with status code " + o.status);
                            }),
                            (o.onerror = s),
                            o.open("GET", t + "?url=" + encodeURIComponent(A) + "&responseType=" + i),
                            "text" !== i && o instanceof XMLHttpRequest && (o.responseType = i),
                            e._options.imageTimeout)
                        ) {
                            var B = e._options.imageTimeout;
                            (o.timeout = B),
                                (o.ontimeout = function () {
                                    return s("Timed out (" + B + "ms) proxying " + r);
                                });
                        }
                        o.send();
                    });
                }),
                A
            );
        })(),
        Re = /^data:image\/svg\+xml/i,
        ye = /^data:image\/.*;base64,/i,
        be = /^data:image\/.*/i,
        Oe = function (A) {
            return Ie.SUPPORT_SVG_DRAWING || !_e(A);
        },
        Se = function (A) {
            return be.test(A);
        },
        Me = function (A) {
            return ye.test(A);
        },
        De = function (A) {
            return "blob" === A.substr(0, 4);
        },
        _e = function (A) {
            return "svg" === A.substr(-3).toLowerCase() || Re.test(A);
        },
        xe = function (A) {
            var e = Pe.CIRCLE,
                t = Ve.FARTHEST_CORNER,
                r = [],
                n = [];
            return (
                XA(A).forEach(function (A, s) {
                    var i = !0;
                    if (
                        (0 === s
                            ? (i = A.reduce(function (A, e) {
                                  if (_A(e))
                                      switch (e.value) {
                                          case "center":
                                              return n.push(qA), !1;
                                          case "top":
                                          case "left":
                                              return n.push(YA), !1;
                                          case "right":
                                          case "bottom":
                                              return n.push(jA), !1;
                                      }
                                  else if (kA(e) || GA(e)) return n.push(e), !1;
                                  return A;
                              }, i))
                            : 1 === s &&
                              (i = A.reduce(function (A, r) {
                                  if (_A(r))
                                      switch (r.value) {
                                          case "circle":
                                              return (e = Pe.CIRCLE), !1;
                                          case "ellipse":
                                              return (e = Pe.ELLIPSE), !1;
                                          case "contain":
                                          case "closest-side":
                                              return (t = Ve.CLOSEST_SIDE), !1;
                                          case "farthest-side":
                                              return (t = Ve.FARTHEST_SIDE), !1;
                                          case "closest-corner":
                                              return (t = Ve.CLOSEST_CORNER), !1;
                                          case "cover":
                                          case "farthest-corner":
                                              return (t = Ve.FARTHEST_CORNER), !1;
                                      }
                                  else if (GA(r) || kA(r)) return Array.isArray(t) || (t = []), t.push(r), !1;
                                  return A;
                              }, i)),
                        i)
                    ) {
                        var o = de(A);
                        r.push(o);
                    }
                }),
                { size: t, shape: e, stops: r, position: n, type: ge.RADIAL_GRADIENT }
            );
        };
    !(function (A) {
        (A[(A.URL = 0)] = "URL"), (A[(A.LINEAR_GRADIENT = 1)] = "LINEAR_GRADIENT"), (A[(A.RADIAL_GRADIENT = 2)] = "RADIAL_GRADIENT");
    })(ge || (ge = {}));
    var Pe, Ve;
    !(function (A) {
        (A[(A.CIRCLE = 0)] = "CIRCLE"), (A[(A.ELLIPSE = 1)] = "ELLIPSE");
    })(Pe || (Pe = {})),
        (function (A) {
            (A[(A.CLOSEST_SIDE = 0)] = "CLOSEST_SIDE"), (A[(A.FARTHEST_SIDE = 1)] = "FARTHEST_SIDE"), (A[(A.CLOSEST_CORNER = 2)] = "CLOSEST_CORNER"), (A[(A.FARTHEST_CORNER = 3)] = "FARTHEST_CORNER");
        })(Ve || (Ve = {}));
    var ze,
        Xe = function (A) {
            if (A.type === U.URL_TOKEN) {
                var e = { url: A.value, type: ge.URL };
                return ve.getInstance().addImage(A.value), e;
            }
            if (A.type === U.FUNCTION) {
                var t = Je[A.name];
                if (void 0 === t) throw new Error('Attempting to parse an unsupported image function "' + A.name + '"');
                return t(A.values);
            }
            throw new Error("Unsupported image type");
        },
        Je = {
            "linear-gradient": function (A) {
                var e = re(180),
                    t = [];
                return (
                    XA(A).forEach(function (A, r) {
                        if (0 === r) {
                            var n = A[0];
                            if (n.type === U.IDENT_TOKEN && "to" === n.value) return void (e = te(A));
                            if (ee(n)) return void (e = Ae(n));
                        }
                        var s = de(A);
                        t.push(s);
                    }),
                    { angle: e, stops: t, type: ge.LINEAR_GRADIENT }
                );
            },
            "-moz-linear-gradient": pe,
            "-ms-linear-gradient": pe,
            "-o-linear-gradient": pe,
            "-webkit-linear-gradient": pe,
            "radial-gradient": function (A) {
                var e = Pe.CIRCLE,
                    t = Ve.FARTHEST_CORNER,
                    r = [],
                    n = [];
                return (
                    XA(A).forEach(function (A, s) {
                        var i = !0;
                        if (0 === s) {
                            var o = !1;
                            i = A.reduce(function (A, r) {
                                if (o)
                                    if (_A(r))
                                        switch (r.value) {
                                            case "center":
                                                return n.push(qA), A;
                                            case "top":
                                            case "left":
                                                return n.push(YA), A;
                                            case "right":
                                            case "bottom":
                                                return n.push(jA), A;
                                        }
                                    else (kA(r) || GA(r)) && n.push(r);
                                else if (_A(r))
                                    switch (r.value) {
                                        case "circle":
                                            return (e = Pe.CIRCLE), !1;
                                        case "ellipse":
                                            return (e = Pe.ELLIPSE), !1;
                                        case "at":
                                            return (o = !0), !1;
                                        case "closest-side":
                                            return (t = Ve.CLOSEST_SIDE), !1;
                                        case "cover":
                                        case "farthest-side":
                                            return (t = Ve.FARTHEST_SIDE), !1;
                                        case "contain":
                                        case "closest-corner":
                                            return (t = Ve.CLOSEST_CORNER), !1;
                                        case "farthest-corner":
                                            return (t = Ve.FARTHEST_CORNER), !1;
                                    }
                                else if (GA(r) || kA(r)) return Array.isArray(t) || (t = []), t.push(r), !1;
                                return A;
                            }, i);
                        }
                        if (i) {
                            var B = de(A);
                            r.push(B);
                        }
                    }),
                    { size: t, shape: e, stops: r, position: n, type: ge.RADIAL_GRADIENT }
                );
            },
            "-moz-radial-gradient": xe,
            "-ms-radial-gradient": xe,
            "-o-radial-gradient": xe,
            "-webkit-radial-gradient": xe,
            "-webkit-gradient": function (A) {
                var e = re(180),
                    t = [],
                    r = ge.LINEAR_GRADIENT,
                    n = Pe.CIRCLE,
                    s = Ve.FARTHEST_CORNER;
                return (
                    XA(A).forEach(function (A, e) {
                        var n = A[0];
                        if (0 === e) {
                            if (_A(n) && "linear" === n.value) return void (r = ge.LINEAR_GRADIENT);
                            if (_A(n) && "radial" === n.value) return void (r = ge.RADIAL_GRADIENT);
                        }
                        if (n.type === U.FUNCTION)
                            if ("from" === n.name) {
                                var s = ne(n.values[0]);
                                t.push({ stop: YA, color: s });
                            } else if ("to" === n.name) (s = ne(n.values[0])), t.push({ stop: jA, color: s });
                            else if ("color-stop" === n.name) {
                                var i = n.values.filter(zA);
                                if (2 === i.length) {
                                    s = ne(i[1]);
                                    var o = i[0];
                                    DA(o) && t.push({ stop: { type: U.PERCENTAGE_TOKEN, number: 100 * o.number, flags: o.flags }, color: s });
                                }
                            }
                    }),
                    r === ge.LINEAR_GRADIENT ? { angle: (e + re(180)) % re(360), stops: t, type: r } : { size: s, shape: n, stops: t, position: [], type: r }
                );
            },
        },
        Ge = {
            name: "background-image",
            initialValue: "none",
            type: le.LIST,
            prefix: !1,
            parse: function (A) {
                if (0 === A.length) return [];
                var e = A[0];
                return e.type === U.IDENT_TOKEN && "none" === e.value ? [] : A.filter(zA).map(Xe);
            },
        },
        ke = {
            name: "background-origin",
            initialValue: "border-box",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return A.map(function (A) {
                    if (_A(A))
                        switch (A.value) {
                            case "padding-box":
                                return 1;
                            case "content-box":
                                return 2;
                        }
                    return 0;
                });
            },
        },
        We = {
            name: "background-position",
            initialValue: "0% 0%",
            type: le.LIST,
            prefix: !1,
            parse: function (A) {
                return XA(A)
                    .map(function (A) {
                        return A.filter(kA);
                    })
                    .map(WA);
            },
        };
    !(function (A) {
        (A[(A.REPEAT = 0)] = "REPEAT"), (A[(A.NO_REPEAT = 1)] = "NO_REPEAT"), (A[(A.REPEAT_X = 2)] = "REPEAT_X"), (A[(A.REPEAT_Y = 3)] = "REPEAT_Y");
    })(ze || (ze = {}));
    var Ye,
        qe = {
            name: "background-repeat",
            initialValue: "repeat",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return XA(A)
                    .map(function (A) {
                        return A.filter(_A)
                            .map(function (A) {
                                return A.value;
                            })
                            .join(" ");
                    })
                    .map(je);
            },
        },
        je = function (A) {
            switch (A) {
                case "no-repeat":
                    return ze.NO_REPEAT;
                case "repeat-x":
                case "repeat no-repeat":
                    return ze.REPEAT_X;
                case "repeat-y":
                case "no-repeat repeat":
                    return ze.REPEAT_Y;
                case "repeat":
                default:
                    return ze.REPEAT;
            }
        };
    !(function (A) {
        (A.AUTO = "auto"), (A.CONTAIN = "contain"), (A.COVER = "cover");
    })(Ye || (Ye = {}));
    var Ze,
        $e = {
            name: "background-size",
            initialValue: "0",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return XA(A).map(function (A) {
                    return A.filter(At);
                });
            },
        },
        At = function (A) {
            return _A(A) || kA(A);
        },
        et = function (A) {
            return { name: "border-" + A + "-color", initialValue: "transparent", prefix: !1, type: le.TYPE_VALUE, format: "color" };
        },
        tt = et("top"),
        rt = et("right"),
        nt = et("bottom"),
        st = et("left"),
        it = function (A) {
            return {
                name: "border-radius-" + A,
                initialValue: "0 0",
                prefix: !1,
                type: le.LIST,
                parse: function (A) {
                    return WA(A.filter(kA));
                },
            };
        },
        ot = it("top-left"),
        Bt = it("top-right"),
        at = it("bottom-right"),
        ct = it("bottom-left");
    !(function (A) {
        (A[(A.NONE = 0)] = "NONE"), (A[(A.SOLID = 1)] = "SOLID");
    })(Ze || (Ze = {}));
    var lt,
        Qt = function (A) {
            return {
                name: "border-" + A + "-style",
                initialValue: "solid",
                prefix: !1,
                type: le.IDENT_VALUE,
                parse: function (A) {
                    switch (A) {
                        case "none":
                            return Ze.NONE;
                    }
                    return Ze.SOLID;
                },
            };
        },
        ut = Qt("top"),
        wt = Qt("right"),
        ht = Qt("bottom"),
        gt = Qt("left"),
        Ut = function (A) {
            return {
                name: "border-" + A + "-width",
                initialValue: "0",
                type: le.VALUE,
                prefix: !1,
                parse: function (A) {
                    return MA(A) ? A.number : 0;
                },
            };
        },
        Ct = Ut("top"),
        dt = Ut("right"),
        Et = Ut("bottom"),
        Ft = Ut("left"),
        ft = { name: "color", initialValue: "transparent", prefix: !1, type: le.TYPE_VALUE, format: "color" },
        Ht = {
            name: "display",
            initialValue: "inline-block",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return A.filter(_A).reduce(function (A, e) {
                    return A | pt(e.value);
                }, 0);
            },
        },
        pt = function (A) {
            switch (A) {
                case "block":
                    return 2;
                case "inline":
                    return 4;
                case "run-in":
                    return 8;
                case "flow":
                    return 16;
                case "flow-root":
                    return 32;
                case "table":
                    return 64;
                case "flex":
                case "-webkit-flex":
                    return 128;
                case "grid":
                    return 256;
                case "ruby":
                    return 512;
                case "subgrid":
                    return 1024;
                case "list-item":
                    return 2048;
                case "table-row-group":
                    return 4096;
                case "table-header-group":
                    return 8192;
                case "table-footer-group":
                    return 16384;
                case "table-row":
                    return 32768;
                case "table-cell":
                    return 65536;
                case "table-column-group":
                    return 131072;
                case "table-column":
                    return 262144;
                case "table-caption":
                    return 524288;
                case "ruby-base":
                    return 1048576;
                case "ruby-text":
                    return 2097152;
                case "ruby-base-container":
                    return 4194304;
                case "ruby-text-container":
                    return 8388608;
                case "contents":
                    return 16777216;
                case "inline-block":
                    return 33554432;
                case "inline-list-item":
                    return 67108864;
                case "inline-table":
                    return 134217728;
                case "inline-flex":
                    return 268435456;
                case "inline-grid":
                    return 536870912;
            }
            return 0;
        };
    !(function (A) {
        (A[(A.NONE = 0)] = "NONE"), (A[(A.LEFT = 1)] = "LEFT"), (A[(A.RIGHT = 2)] = "RIGHT"), (A[(A.INLINE_START = 3)] = "INLINE_START"), (A[(A.INLINE_END = 4)] = "INLINE_END");
    })(lt || (lt = {}));
    var Nt,
        Kt = {
            name: "float",
            initialValue: "none",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "left":
                        return lt.LEFT;
                    case "right":
                        return lt.RIGHT;
                    case "inline-start":
                        return lt.INLINE_START;
                    case "inline-end":
                        return lt.INLINE_END;
                }
                return lt.NONE;
            },
        },
        mt = {
            name: "letter-spacing",
            initialValue: "0",
            prefix: !1,
            type: le.VALUE,
            parse: function (A) {
                return A.type === U.IDENT_TOKEN && "normal" === A.value ? 0 : A.type === U.NUMBER_TOKEN ? A.number : A.type === U.DIMENSION_TOKEN ? A.number : 0;
            },
        };
    !(function (A) {
        (A.NORMAL = "normal"), (A.STRICT = "strict");
    })(Nt || (Nt = {}));
    var It,
        Tt = {
            name: "line-break",
            initialValue: "normal",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "strict":
                        return Nt.STRICT;
                    case "normal":
                    default:
                        return Nt.NORMAL;
                }
            },
        },
        vt = { name: "line-height", initialValue: "normal", prefix: !1, type: le.TOKEN_VALUE },
        Lt = {
            name: "list-style-image",
            initialValue: "none",
            type: le.VALUE,
            prefix: !1,
            parse: function (A) {
                return A.type === U.IDENT_TOKEN && "none" === A.value ? null : Xe(A);
            },
        };
    !(function (A) {
        (A[(A.INSIDE = 0)] = "INSIDE"), (A[(A.OUTSIDE = 1)] = "OUTSIDE");
    })(It || (It = {}));
    var Rt,
        yt = {
            name: "list-style-position",
            initialValue: "outside",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "inside":
                        return It.INSIDE;
                    case "outside":
                    default:
                        return It.OUTSIDE;
                }
            },
        };
    !(function (A) {
        (A[(A.NONE = -1)] = "NONE"),
            (A[(A.DISC = 0)] = "DISC"),
            (A[(A.CIRCLE = 1)] = "CIRCLE"),
            (A[(A.SQUARE = 2)] = "SQUARE"),
            (A[(A.DECIMAL = 3)] = "DECIMAL"),
            (A[(A.CJK_DECIMAL = 4)] = "CJK_DECIMAL"),
            (A[(A.DECIMAL_LEADING_ZERO = 5)] = "DECIMAL_LEADING_ZERO"),
            (A[(A.LOWER_ROMAN = 6)] = "LOWER_ROMAN"),
            (A[(A.UPPER_ROMAN = 7)] = "UPPER_ROMAN"),
            (A[(A.LOWER_GREEK = 8)] = "LOWER_GREEK"),
            (A[(A.LOWER_ALPHA = 9)] = "LOWER_ALPHA"),
            (A[(A.UPPER_ALPHA = 10)] = "UPPER_ALPHA"),
            (A[(A.ARABIC_INDIC = 11)] = "ARABIC_INDIC"),
            (A[(A.ARMENIAN = 12)] = "ARMENIAN"),
            (A[(A.BENGALI = 13)] = "BENGALI"),
            (A[(A.CAMBODIAN = 14)] = "CAMBODIAN"),
            (A[(A.CJK_EARTHLY_BRANCH = 15)] = "CJK_EARTHLY_BRANCH"),
            (A[(A.CJK_HEAVENLY_STEM = 16)] = "CJK_HEAVENLY_STEM"),
            (A[(A.CJK_IDEOGRAPHIC = 17)] = "CJK_IDEOGRAPHIC"),
            (A[(A.DEVANAGARI = 18)] = "DEVANAGARI"),
            (A[(A.ETHIOPIC_NUMERIC = 19)] = "ETHIOPIC_NUMERIC"),
            (A[(A.GEORGIAN = 20)] = "GEORGIAN"),
            (A[(A.GUJARATI = 21)] = "GUJARATI"),
            (A[(A.GURMUKHI = 22)] = "GURMUKHI"),
            (A[(A.HEBREW = 22)] = "HEBREW"),
            (A[(A.HIRAGANA = 23)] = "HIRAGANA"),
            (A[(A.HIRAGANA_IROHA = 24)] = "HIRAGANA_IROHA"),
            (A[(A.JAPANESE_FORMAL = 25)] = "JAPANESE_FORMAL"),
            (A[(A.JAPANESE_INFORMAL = 26)] = "JAPANESE_INFORMAL"),
            (A[(A.KANNADA = 27)] = "KANNADA"),
            (A[(A.KATAKANA = 28)] = "KATAKANA"),
            (A[(A.KATAKANA_IROHA = 29)] = "KATAKANA_IROHA"),
            (A[(A.KHMER = 30)] = "KHMER"),
            (A[(A.KOREAN_HANGUL_FORMAL = 31)] = "KOREAN_HANGUL_FORMAL"),
            (A[(A.KOREAN_HANJA_FORMAL = 32)] = "KOREAN_HANJA_FORMAL"),
            (A[(A.KOREAN_HANJA_INFORMAL = 33)] = "KOREAN_HANJA_INFORMAL"),
            (A[(A.LAO = 34)] = "LAO"),
            (A[(A.LOWER_ARMENIAN = 35)] = "LOWER_ARMENIAN"),
            (A[(A.MALAYALAM = 36)] = "MALAYALAM"),
            (A[(A.MONGOLIAN = 37)] = "MONGOLIAN"),
            (A[(A.MYANMAR = 38)] = "MYANMAR"),
            (A[(A.ORIYA = 39)] = "ORIYA"),
            (A[(A.PERSIAN = 40)] = "PERSIAN"),
            (A[(A.SIMP_CHINESE_FORMAL = 41)] = "SIMP_CHINESE_FORMAL"),
            (A[(A.SIMP_CHINESE_INFORMAL = 42)] = "SIMP_CHINESE_INFORMAL"),
            (A[(A.TAMIL = 43)] = "TAMIL"),
            (A[(A.TELUGU = 44)] = "TELUGU"),
            (A[(A.THAI = 45)] = "THAI"),
            (A[(A.TIBETAN = 46)] = "TIBETAN"),
            (A[(A.TRAD_CHINESE_FORMAL = 47)] = "TRAD_CHINESE_FORMAL"),
            (A[(A.TRAD_CHINESE_INFORMAL = 48)] = "TRAD_CHINESE_INFORMAL"),
            (A[(A.UPPER_ARMENIAN = 49)] = "UPPER_ARMENIAN"),
            (A[(A.DISCLOSURE_OPEN = 50)] = "DISCLOSURE_OPEN"),
            (A[(A.DISCLOSURE_CLOSED = 51)] = "DISCLOSURE_CLOSED");
    })(Rt || (Rt = {}));
    var bt,
        Ot = {
            name: "list-style-type",
            initialValue: "none",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "disc":
                        return Rt.DISC;
                    case "circle":
                        return Rt.CIRCLE;
                    case "square":
                        return Rt.SQUARE;
                    case "decimal":
                        return Rt.DECIMAL;
                    case "cjk-decimal":
                        return Rt.CJK_DECIMAL;
                    case "decimal-leading-zero":
                        return Rt.DECIMAL_LEADING_ZERO;
                    case "lower-roman":
                        return Rt.LOWER_ROMAN;
                    case "upper-roman":
                        return Rt.UPPER_ROMAN;
                    case "lower-greek":
                        return Rt.LOWER_GREEK;
                    case "lower-alpha":
                        return Rt.LOWER_ALPHA;
                    case "upper-alpha":
                        return Rt.UPPER_ALPHA;
                    case "arabic-indic":
                        return Rt.ARABIC_INDIC;
                    case "armenian":
                        return Rt.ARMENIAN;
                    case "bengali":
                        return Rt.BENGALI;
                    case "cambodian":
                        return Rt.CAMBODIAN;
                    case "cjk-earthly-branch":
                        return Rt.CJK_EARTHLY_BRANCH;
                    case "cjk-heavenly-stem":
                        return Rt.CJK_HEAVENLY_STEM;
                    case "cjk-ideographic":
                        return Rt.CJK_IDEOGRAPHIC;
                    case "devanagari":
                        return Rt.DEVANAGARI;
                    case "ethiopic-numeric":
                        return Rt.ETHIOPIC_NUMERIC;
                    case "georgian":
                        return Rt.GEORGIAN;
                    case "gujarati":
                        return Rt.GUJARATI;
                    case "gurmukhi":
                        return Rt.GURMUKHI;
                    case "hebrew":
                        return Rt.HEBREW;
                    case "hiragana":
                        return Rt.HIRAGANA;
                    case "hiragana-iroha":
                        return Rt.HIRAGANA_IROHA;
                    case "japanese-formal":
                        return Rt.JAPANESE_FORMAL;
                    case "japanese-informal":
                        return Rt.JAPANESE_INFORMAL;
                    case "kannada":
                        return Rt.KANNADA;
                    case "katakana":
                        return Rt.KATAKANA;
                    case "katakana-iroha":
                        return Rt.KATAKANA_IROHA;
                    case "khmer":
                        return Rt.KHMER;
                    case "korean-hangul-formal":
                        return Rt.KOREAN_HANGUL_FORMAL;
                    case "korean-hanja-formal":
                        return Rt.KOREAN_HANJA_FORMAL;
                    case "korean-hanja-informal":
                        return Rt.KOREAN_HANJA_INFORMAL;
                    case "lao":
                        return Rt.LAO;
                    case "lower-armenian":
                        return Rt.LOWER_ARMENIAN;
                    case "malayalam":
                        return Rt.MALAYALAM;
                    case "mongolian":
                        return Rt.MONGOLIAN;
                    case "myanmar":
                        return Rt.MYANMAR;
                    case "oriya":
                        return Rt.ORIYA;
                    case "persian":
                        return Rt.PERSIAN;
                    case "simp-chinese-formal":
                        return Rt.SIMP_CHINESE_FORMAL;
                    case "simp-chinese-informal":
                        return Rt.SIMP_CHINESE_INFORMAL;
                    case "tamil":
                        return Rt.TAMIL;
                    case "telugu":
                        return Rt.TELUGU;
                    case "thai":
                        return Rt.THAI;
                    case "tibetan":
                        return Rt.TIBETAN;
                    case "trad-chinese-formal":
                        return Rt.TRAD_CHINESE_FORMAL;
                    case "trad-chinese-informal":
                        return Rt.TRAD_CHINESE_INFORMAL;
                    case "upper-armenian":
                        return Rt.UPPER_ARMENIAN;
                    case "disclosure-open":
                        return Rt.DISCLOSURE_OPEN;
                    case "disclosure-closed":
                        return Rt.DISCLOSURE_CLOSED;
                    case "none":
                    default:
                        return Rt.NONE;
                }
            },
        },
        St = function (A) {
            return { name: "margin-" + A, initialValue: "0", prefix: !1, type: le.TOKEN_VALUE };
        },
        Mt = St("top"),
        Dt = St("right"),
        _t = St("bottom"),
        xt = St("left");
    !(function (A) {
        (A[(A.VISIBLE = 0)] = "VISIBLE"), (A[(A.HIDDEN = 1)] = "HIDDEN"), (A[(A.SCROLL = 2)] = "SCROLL"), (A[(A.AUTO = 3)] = "AUTO");
    })(bt || (bt = {}));
    var Pt,
        Vt = {
            name: "overflow",
            initialValue: "visible",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return A.filter(_A).map(function (A) {
                    switch (A.value) {
                        case "hidden":
                            return bt.HIDDEN;
                        case "scroll":
                            return bt.SCROLL;
                        case "auto":
                            return bt.AUTO;
                        case "visible":
                        default:
                            return bt.VISIBLE;
                    }
                });
            },
        };
    !(function (A) {
        (A.NORMAL = "normal"), (A.BREAK_WORD = "break-word");
    })(Pt || (Pt = {}));
    var zt,
        Xt = {
            name: "overflow-wrap",
            initialValue: "normal",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "break-word":
                        return Pt.BREAK_WORD;
                    case "normal":
                    default:
                        return Pt.NORMAL;
                }
            },
        },
        Jt = function (A) {
            return { name: "padding-" + A, initialValue: "0", prefix: !1, type: le.TYPE_VALUE, format: "length-percentage" };
        },
        Gt = Jt("top"),
        kt = Jt("right"),
        Wt = Jt("bottom"),
        Yt = Jt("left");
    !(function (A) {
        (A[(A.LEFT = 0)] = "LEFT"), (A[(A.CENTER = 1)] = "CENTER"), (A[(A.RIGHT = 2)] = "RIGHT");
    })(zt || (zt = {}));
    var qt,
        jt = {
            name: "text-align",
            initialValue: "left",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "right":
                        return zt.RIGHT;
                    case "center":
                    case "justify":
                        return zt.CENTER;
                    case "left":
                    default:
                        return zt.LEFT;
                }
            },
        };
    !(function (A) {
        (A[(A.STATIC = 0)] = "STATIC"), (A[(A.RELATIVE = 1)] = "RELATIVE"), (A[(A.ABSOLUTE = 2)] = "ABSOLUTE"), (A[(A.FIXED = 3)] = "FIXED"), (A[(A.STICKY = 4)] = "STICKY");
    })(qt || (qt = {}));
    var Zt,
        $t = {
            name: "position",
            initialValue: "static",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "relative":
                        return qt.RELATIVE;
                    case "absolute":
                        return qt.ABSOLUTE;
                    case "fixed":
                        return qt.FIXED;
                    case "sticky":
                        return qt.STICKY;
                }
                return qt.STATIC;
            },
        },
        Ar = {
            name: "text-shadow",
            initialValue: "none",
            type: le.LIST,
            prefix: !1,
            parse: function (A) {
                return 1 === A.length && PA(A[0], "none")
                    ? []
                    : XA(A).map(function (A) {
                          for (var e = { color: he.TRANSPARENT, offsetX: YA, offsetY: YA, blur: YA }, t = 0, r = 0; r < A.length; r++) {
                              var n = A[r];
                              GA(n) ? (0 === t ? (e.offsetX = n) : 1 === t ? (e.offsetY = n) : (e.blur = n), t++) : (e.color = ne(n));
                          }
                          return e;
                      });
            },
        };
    !(function (A) {
        (A[(A.NONE = 0)] = "NONE"), (A[(A.LOWERCASE = 1)] = "LOWERCASE"), (A[(A.UPPERCASE = 2)] = "UPPERCASE"), (A[(A.CAPITALIZE = 3)] = "CAPITALIZE");
    })(Zt || (Zt = {}));
    var er,
        tr = {
            name: "text-transform",
            initialValue: "none",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "uppercase":
                        return Zt.UPPERCASE;
                    case "lowercase":
                        return Zt.LOWERCASE;
                    case "capitalize":
                        return Zt.CAPITALIZE;
                }
                return Zt.NONE;
            },
        },
        rr = {
            name: "transform",
            initialValue: "none",
            prefix: !0,
            type: le.VALUE,
            parse: function (A) {
                if (A.type === U.IDENT_TOKEN && "none" === A.value) return null;
                if (A.type === U.FUNCTION) {
                    var e = nr[A.name];
                    if (void 0 === e) throw new Error('Attempting to parse an unsupported transform function "' + A.name + '"');
                    return e(A.values);
                }
                return null;
            },
        },
        nr = {
            matrix: function (A) {
                var e = A.filter(function (A) {
                    return A.type === U.NUMBER_TOKEN;
                }).map(function (A) {
                    return A.number;
                });
                return 6 === e.length ? e : null;
            },
            matrix3d: function (A) {
                var e = A.filter(function (A) {
                        return A.type === U.NUMBER_TOKEN;
                    }).map(function (A) {
                        return A.number;
                    }),
                    t = e[0],
                    r = e[1],
                    n = (e[2], e[3], e[4]),
                    s = e[5],
                    i = (e[6], e[7], e[8], e[9], e[10], e[11], e[12]),
                    o = e[13];
                e[14], e[15];
                return 16 === e.length ? [t, r, n, s, i, o] : null;
            },
        },
        sr = { type: U.PERCENTAGE_TOKEN, number: 50, flags: 4 },
        ir = [sr, sr],
        or = {
            name: "transform-origin",
            initialValue: "50% 50%",
            prefix: !0,
            type: le.LIST,
            parse: function (A) {
                var e = A.filter(kA);
                return 2 !== e.length ? ir : [e[0], e[1]];
            },
        };
    !(function (A) {
        (A[(A.VISIBLE = 0)] = "VISIBLE"), (A[(A.HIDDEN = 1)] = "HIDDEN"), (A[(A.COLLAPSE = 2)] = "COLLAPSE");
    })(er || (er = {}));
    var Br,
        ar = {
            name: "visible",
            initialValue: "none",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "hidden":
                        return er.HIDDEN;
                    case "collapse":
                        return er.COLLAPSE;
                    case "visible":
                    default:
                        return er.VISIBLE;
                }
            },
        };
    !(function (A) {
        (A.NORMAL = "normal"), (A.BREAK_ALL = "break-all"), (A.KEEP_ALL = "keep-all");
    })(Br || (Br = {}));
    var cr,
        lr = {
            name: "word-break",
            initialValue: "normal",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "break-all":
                        return Br.BREAK_ALL;
                    case "keep-all":
                        return Br.KEEP_ALL;
                    case "normal":
                    default:
                        return Br.NORMAL;
                }
            },
        },
        Qr = {
            name: "z-index",
            initialValue: "auto",
            prefix: !1,
            type: le.VALUE,
            parse: function (A) {
                if (A.type === U.IDENT_TOKEN) return { auto: !0, order: 0 };
                if (DA(A)) return { auto: !1, order: A.number };
                throw new Error("Invalid z-index number parsed");
            },
        },
        ur = {
            name: "opacity",
            initialValue: "1",
            type: le.VALUE,
            prefix: !1,
            parse: function (A) {
                return DA(A) ? A.number : 1;
            },
        },
        wr = { name: "text-decoration-color", initialValue: "transparent", prefix: !1, type: le.TYPE_VALUE, format: "color" },
        hr = {
            name: "text-decoration-line",
            initialValue: "none",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return A.filter(_A)
                    .map(function (A) {
                        switch (A.value) {
                            case "underline":
                                return 1;
                            case "overline":
                                return 2;
                            case "line-through":
                                return 3;
                            case "none":
                                return 4;
                        }
                        return 0;
                    })
                    .filter(function (A) {
                        return 0 !== A;
                    });
            },
        },
        gr = {
            name: "font-family",
            initialValue: "",
            prefix: !1,
            type: le.LIST,
            parse: function (A) {
                return A.filter(Ur).map(function (A) {
                    return A.value;
                });
            },
        },
        Ur = function (A) {
            return A.type === U.STRING_TOKEN || A.type === U.IDENT_TOKEN;
        },
        Cr = { name: "font-size", initialValue: "0", prefix: !1, type: le.TYPE_VALUE, format: "length" },
        dr = {
            name: "font-weight",
            initialValue: "normal",
            type: le.VALUE,
            prefix: !1,
            parse: function (A) {
                if (DA(A)) return A.number;
                if (_A(A))
                    switch (A.value) {
                        case "bold":
                            return 700;
                        case "normal":
                        default:
                            return 400;
                    }
                return 400;
            },
        },
        Er = {
            name: "font-variant",
            initialValue: "none",
            type: le.LIST,
            prefix: !1,
            parse: function (A) {
                return A.filter(_A).map(function (A) {
                    return A.value;
                });
            },
        };
    !(function (A) {
        (A.NORMAL = "normal"), (A.ITALIC = "italic"), (A.OBLIQUE = "oblique");
    })(cr || (cr = {}));
    var Fr,
        fr = {
            name: "font-style",
            initialValue: "normal",
            prefix: !1,
            type: le.IDENT_VALUE,
            parse: function (A) {
                switch (A) {
                    case "oblique":
                        return cr.OBLIQUE;
                    case "italic":
                        return cr.ITALIC;
                    case "normal":
                    default:
                        return cr.NORMAL;
                }
            },
        },
        Hr = function (A, e) {
            return 0 != (A & e);
        },
        pr = {
            name: "content",
            initialValue: "none",
            type: le.LIST,
            prefix: !1,
            parse: function (A) {
                if (0 === A.length) return [];
                var e = A[0];
                return e.type === U.IDENT_TOKEN && "none" === e.value ? [] : A;
            },
        },
        Nr = {
            name: "counter-increment",
            initialValue: "none",
            prefix: !0,
            type: le.LIST,
            parse: function (A) {
                if (0 === A.length) return null;
                var e = A[0];
                if (e.type === U.IDENT_TOKEN && "none" === e.value) return null;
                for (var t = [], r = A.filter(VA), n = 0; n < r.length; n++) {
                    var s = r[n],
                        i = r[n + 1];
                    if (s.type === U.IDENT_TOKEN) {
                        var o = i && DA(i) ? i.number : 1;
                        t.push({ counter: s.value, increment: o });
                    }
                }
                return t;
            },
        },
        Kr = {
            name: "counter-reset",
            initialValue: "none",
            prefix: !0,
            type: le.LIST,
            parse: function (A) {
                if (0 === A.length) return [];
                for (var e = [], t = A.filter(VA), r = 0; r < t.length; r++) {
                    var n = t[r],
                        s = t[r + 1];
                    if (_A(n) && "none" !== n.value) {
                        var i = s && DA(s) ? s.number : 0;
                        e.push({ counter: n.value, reset: i });
                    }
                }
                return e;
            },
        },
        mr = {
            name: "quotes",
            initialValue: "none",
            prefix: !0,
            type: le.LIST,
            parse: function (A) {
                if (0 === A.length) return null;
                var e = A[0];
                if (e.type === U.IDENT_TOKEN && "none" === e.value) return null;
                var t = [],
                    r = A.filter(xA);
                if (r.length % 2 != 0) return null;
                for (var n = 0; n < r.length; n += 2) {
                    var s = r[n].value,
                        i = r[n + 1].value;
                    t.push({ open: s, close: i });
                }
                return t;
            },
        },
        Ir = function (A, e, t) {
            if (!A) return "";
            var r = A[Math.min(e, A.length - 1)];
            return r ? (t ? r.open : r.close) : "";
        },
        Tr = {
            name: "box-shadow",
            initialValue: "none",
            type: le.LIST,
            prefix: !1,
            parse: function (A) {
                return 1 === A.length && PA(A[0], "none")
                    ? []
                    : XA(A).map(function (A) {
                          for (var e = { color: 255, offsetX: YA, offsetY: YA, blur: YA, spread: YA, inset: !1 }, t = 0, r = 0; r < A.length; r++) {
                              var n = A[r];
                              PA(n, "inset") ? (e.inset = !0) : GA(n) ? (0 === t ? (e.offsetX = n) : 1 === t ? (e.offsetY = n) : 2 === t ? (e.blur = n) : (e.spread = n), t++) : (e.color = ne(n));
                          }
                          return e;
                      });
            },
        },
        vr = (function () {
            function A(A) {
                (this.backgroundClip = yr(Ue, A.backgroundClip)),
                    (this.backgroundColor = yr(Ce, A.backgroundColor)),
                    (this.backgroundImage = yr(Ge, A.backgroundImage)),
                    (this.backgroundOrigin = yr(ke, A.backgroundOrigin)),
                    (this.backgroundPosition = yr(We, A.backgroundPosition)),
                    (this.backgroundRepeat = yr(qe, A.backgroundRepeat)),
                    (this.backgroundSize = yr($e, A.backgroundSize)),
                    (this.borderTopColor = yr(tt, A.borderTopColor)),
                    (this.borderRightColor = yr(rt, A.borderRightColor)),
                    (this.borderBottomColor = yr(nt, A.borderBottomColor)),
                    (this.borderLeftColor = yr(st, A.borderLeftColor)),
                    (this.borderTopLeftRadius = yr(ot, A.borderTopLeftRadius)),
                    (this.borderTopRightRadius = yr(Bt, A.borderTopRightRadius)),
                    (this.borderBottomRightRadius = yr(at, A.borderBottomRightRadius)),
                    (this.borderBottomLeftRadius = yr(ct, A.borderBottomLeftRadius)),
                    (this.borderTopStyle = yr(ut, A.borderTopStyle)),
                    (this.borderRightStyle = yr(wt, A.borderRightStyle)),
                    (this.borderBottomStyle = yr(ht, A.borderBottomStyle)),
                    (this.borderLeftStyle = yr(gt, A.borderLeftStyle)),
                    (this.borderTopWidth = yr(Ct, A.borderTopWidth)),
                    (this.borderRightWidth = yr(dt, A.borderRightWidth)),
                    (this.borderBottomWidth = yr(Et, A.borderBottomWidth)),
                    (this.borderLeftWidth = yr(Ft, A.borderLeftWidth)),
                    (this.boxShadow = yr(Tr, A.boxShadow)),
                    (this.color = yr(ft, A.color)),
                    (this.display = yr(Ht, A.display)),
                    (this.float = yr(Kt, A.cssFloat)),
                    (this.fontFamily = yr(gr, A.fontFamily)),
                    (this.fontSize = yr(Cr, A.fontSize)),
                    (this.fontStyle = yr(fr, A.fontStyle)),
                    (this.fontVariant = yr(Er, A.fontVariant)),
                    (this.fontWeight = yr(dr, A.fontWeight)),
                    (this.letterSpacing = yr(mt, A.letterSpacing)),
                    (this.lineBreak = yr(Tt, A.lineBreak)),
                    (this.lineHeight = yr(vt, A.lineHeight)),
                    (this.listStyleImage = yr(Lt, A.listStyleImage)),
                    (this.listStylePosition = yr(yt, A.listStylePosition)),
                    (this.listStyleType = yr(Ot, A.listStyleType)),
                    (this.marginTop = yr(Mt, A.marginTop)),
                    (this.marginRight = yr(Dt, A.marginRight)),
                    (this.marginBottom = yr(_t, A.marginBottom)),
                    (this.marginLeft = yr(xt, A.marginLeft)),
                    (this.opacity = yr(ur, A.opacity));
                var e = yr(Vt, A.overflow);
                (this.overflowX = e[0]),
                    (this.overflowY = e[e.length > 1 ? 1 : 0]),
                    (this.overflowWrap = yr(Xt, A.overflowWrap)),
                    (this.paddingTop = yr(Gt, A.paddingTop)),
                    (this.paddingRight = yr(kt, A.paddingRight)),
                    (this.paddingBottom = yr(Wt, A.paddingBottom)),
                    (this.paddingLeft = yr(Yt, A.paddingLeft)),
                    (this.position = yr($t, A.position)),
                    (this.textAlign = yr(jt, A.textAlign)),
                    (this.textDecorationColor = yr(wr, A.textDecorationColor || A.color)),
                    (this.textDecorationLine = yr(hr, A.textDecorationLine)),
                    (this.textShadow = yr(Ar, A.textShadow)),
                    (this.textTransform = yr(tr, A.textTransform)),
                    (this.transform = yr(rr, A.transform)),
                    (this.transformOrigin = yr(or, A.transformOrigin)),
                    (this.visibility = yr(ar, A.visibility)),
                    (this.wordBreak = yr(lr, A.wordBreak)),
                    (this.zIndex = yr(Qr, A.zIndex));
            }
            return (
                (A.prototype.isVisible = function () {
                    return this.display > 0 && this.opacity > 0 && this.visibility === er.VISIBLE;
                }),
                (A.prototype.isTransparent = function () {
                    return se(this.backgroundColor);
                }),
                (A.prototype.isTransformed = function () {
                    return null !== this.transform;
                }),
                (A.prototype.isPositioned = function () {
                    return this.position !== qt.STATIC;
                }),
                (A.prototype.isPositionedWithZIndex = function () {
                    return this.isPositioned() && !this.zIndex.auto;
                }),
                (A.prototype.isFloating = function () {
                    return this.float !== lt.NONE;
                }),
                (A.prototype.isInlineLevel = function () {
                    return Hr(this.display, 4) || Hr(this.display, 33554432) || Hr(this.display, 268435456) || Hr(this.display, 536870912) || Hr(this.display, 67108864) || Hr(this.display, 134217728);
                }),
                A
            );
        })(),
        Lr = (function () {
            return function (A) {
                (this.content = yr(pr, A.content)), (this.quotes = yr(mr, A.quotes));
            };
        })(),
        Rr = (function () {
            return function (A) {
                (this.counterIncrement = yr(Nr, A.counterIncrement)), (this.counterReset = yr(Kr, A.counterReset));
            };
        })(),
        yr = function (A, e) {
            var t = new OA(),
                r = null != e ? e.toString() : A.initialValue;
            t.write(r);
            var n = new SA(t.read());
            switch (A.type) {
                case le.IDENT_VALUE:
                    var s = n.parseComponentValue();
                    return A.parse(_A(s) ? s.value : A.initialValue);
                case le.VALUE:
                    return A.parse(n.parseComponentValue());
                case le.LIST:
                    return A.parse(n.parseComponentValues());
                case le.TOKEN_VALUE:
                    return n.parseComponentValue();
                case le.TYPE_VALUE:
                    switch (A.format) {
                        case "angle":
                            return Ae(n.parseComponentValue());
                        case "color":
                            return ne(n.parseComponentValue());
                        case "image":
                            return Xe(n.parseComponentValue());
                        case "length":
                            var i = n.parseComponentValue();
                            return GA(i) ? i : YA;
                        case "length-percentage":
                            var o = n.parseComponentValue();
                            return kA(o) ? o : YA;
                    }
            }
            throw new Error("Attempting to parse unsupported css format type " + A.format);
        },
        br = (function () {
            return function (A) {
                (this.styles = new vr(window.getComputedStyle(A, null))), (this.textNodes = []), (this.elements = []), null !== this.styles.transform && Un(A) && (A.style.transform = "none"), (this.bounds = l(A)), (this.flags = 0);
            };
        })(),
        Or = (function () {
            return function (A, e) {
                (this.text = A), (this.bounds = e);
            };
        })(),
        Sr = function (A, e, t) {
            var r = _r(A, e),
                n = [],
                s = 0;
            return (
                r.forEach(function (A) {
                    if (e.textDecorationLine.length || A.trim().length > 0)
                        if (Ie.SUPPORT_RANGE_BOUNDS) n.push(new Or(A, Dr(t, s, A.length)));
                        else {
                            var r = t.splitText(A.length);
                            n.push(new Or(A, Mr(t))), (t = r);
                        }
                    else Ie.SUPPORT_RANGE_BOUNDS || (t = t.splitText(A.length));
                    s += A.length;
                }),
                n
            );
        },
        Mr = function (A) {
            var e = A.ownerDocument;
            if (e) {
                var t = e.createElement("html2canvaswrapper");
                t.appendChild(A.cloneNode(!0));
                var r = A.parentNode;
                if (r) {
                    r.replaceChild(t, A);
                    var n = l(t);
                    return t.firstChild && r.replaceChild(t.firstChild, t), n;
                }
            }
            return new c(0, 0, 0, 0);
        },
        Dr = function (A, e, t) {
            var r = A.ownerDocument;
            if (!r) throw new Error("Node has no owner document");
            var n = r.createRange();
            return n.setStart(A, e), n.setEnd(A, e + t), c.fromClientRect(n.getBoundingClientRect());
        },
        _r = function (A, e) {
            return 0 !== e.letterSpacing
                ? Q(A).map(function (A) {
                      return u(A);
                  })
                : xr(A, e);
        },
        xr = function (A, e) {
            for (
                var t,
                    r = (function (A, e) {
                        var t = Q(A),
                            r = tA(t, e),
                            n = r[0],
                            s = r[1],
                            i = r[2],
                            o = t.length,
                            B = 0,
                            a = 0;
                        return {
                            next: function () {
                                if (a >= o) return { done: !0, value: null };
                                for (var A = "??"; a < o && "??" === (A = eA(t, s, n, ++a, i)); );
                                if ("??" !== A || a === o) {
                                    var e = new rA(t, A, B, a);
                                    return (B = a), { value: e, done: !1 };
                                }
                                return { done: !0, value: null };
                            },
                        };
                    })(A, { lineBreak: e.lineBreak, wordBreak: e.overflowWrap === Pt.BREAK_WORD ? "break-word" : e.wordBreak }),
                    n = [];
                !(t = r.next()).done;

            )
                t.value && n.push(t.value.slice());
            return n;
        },
        Pr = (function () {
            return function (A, e) {
                (this.text = Vr(A.data, e.textTransform)), (this.textBounds = Sr(this.text, e, A));
            };
        })(),
        Vr = function (A, e) {
            switch (e) {
                case Zt.LOWERCASE:
                    return A.toLowerCase();
                case Zt.CAPITALIZE:
                    return A.replace(zr, Xr);
                case Zt.UPPERCASE:
                    return A.toUpperCase();
                default:
                    return A;
            }
        },
        zr = /(^|\s|:|-|\(|\))([a-z])/g,
        Xr = function (A, e, t) {
            return A.length > 0 ? e + t.toUpperCase() : A;
        },
        Jr = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this;
                return (t.src = e.currentSrc || e.src), (t.intrinsicWidth = e.naturalWidth), (t.intrinsicHeight = e.naturalHeight), ve.getInstance().addImage(t.src), t;
            }
            return i(e, A), e;
        })(br),
        Gr = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this;
                return (t.canvas = e), (t.intrinsicWidth = e.width), (t.intrinsicHeight = e.height), t;
            }
            return i(e, A), e;
        })(br),
        kr = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this,
                    r = new XMLSerializer();
                return (t.svg = "data:image/svg+xml," + encodeURIComponent(r.serializeToString(e))), (t.intrinsicWidth = e.width.baseVal.value), (t.intrinsicHeight = e.height.baseVal.value), ve.getInstance().addImage(t.svg), t;
            }
            return i(e, A), e;
        })(br),
        Wr = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this;
                return (t.value = e.value), t;
            }
            return i(e, A), e;
        })(br),
        Yr = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this;
                return (t.start = e.start), (t.reversed = "boolean" == typeof e.reversed && !0 === e.reversed), t;
            }
            return i(e, A), e;
        })(br),
        qr = [{ type: U.DIMENSION_TOKEN, flags: 0, unit: "px", number: 3 }],
        jr = [{ type: U.PERCENTAGE_TOKEN, flags: 0, number: 50 }],
        Zr = function (A) {
            return A.width > A.height ? new c(A.left + (A.width - A.height) / 2, A.top, A.height, A.height) : A.width < A.height ? new c(A.left, A.top + (A.height - A.width) / 2, A.width, A.width) : A;
        },
        $r = function (A) {
            var e = A.type === tn ? new Array(A.value.length + 1).join("???") : A.value;
            return 0 === e.length ? A.placeholder || "" : e;
        },
        An = "checkbox",
        en = "radio",
        tn = "password",
        rn = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this;
                switch (
                    ((t.type = e.type.toLowerCase()),
                    (t.checked = e.checked),
                    (t.value = $r(e)),
                    (t.type !== An && t.type !== en) ||
                        ((t.styles.backgroundColor = 3739148031),
                        (t.styles.borderTopColor = t.styles.borderRightColor = t.styles.borderBottomColor = t.styles.borderLeftColor = 2779096575),
                        (t.styles.borderTopWidth = t.styles.borderRightWidth = t.styles.borderBottomWidth = t.styles.borderLeftWidth = 1),
                        (t.styles.borderTopStyle = t.styles.borderRightStyle = t.styles.borderBottomStyle = t.styles.borderLeftStyle = Ze.SOLID),
                        (t.styles.backgroundClip = [Qe.BORDER_BOX]),
                        (t.styles.backgroundOrigin = [0]),
                        (t.bounds = Zr(t.bounds))),
                    t.type)
                ) {
                    case An:
                        t.styles.borderTopRightRadius = t.styles.borderTopLeftRadius = t.styles.borderBottomRightRadius = t.styles.borderBottomLeftRadius = qr;
                        break;
                    case en:
                        t.styles.borderTopRightRadius = t.styles.borderTopLeftRadius = t.styles.borderBottomRightRadius = t.styles.borderBottomLeftRadius = jr;
                }
                return t;
            }
            return i(e, A), e;
        })(br),
        nn = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this,
                    r = e.options[e.selectedIndex || 0];
                return (t.value = (r && r.text) || ""), t;
            }
            return i(e, A), e;
        })(br),
        sn = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this;
                return (t.value = e.value), t;
            }
            return i(e, A), e;
        })(br),
        on = function (A) {
            return ne(SA.create(A).parseComponentValue());
        },
        Bn = (function (A) {
            function e(e) {
                var t = A.call(this, e) || this;
                (t.src = e.src), (t.width = parseInt(e.width, 10)), (t.height = parseInt(e.height, 10)), (t.backgroundColor = t.styles.backgroundColor);
                try {
                    if (e.contentWindow && e.contentWindow.document && e.contentWindow.document.documentElement) {
                        t.tree = Qn(e.contentWindow.document.documentElement);
                        var r = e.contentWindow.document.documentElement ? on(getComputedStyle(e.contentWindow.document.documentElement).backgroundColor) : he.TRANSPARENT,
                            n = e.contentWindow.document.body ? on(getComputedStyle(e.contentWindow.document.body).backgroundColor) : he.TRANSPARENT;
                        t.backgroundColor = se(r) ? (se(n) ? t.styles.backgroundColor : n) : r;
                    }
                } catch (A) {}
                return t;
            }
            return i(e, A), e;
        })(br),
        an = ["OL", "UL", "MENU"],
        cn = function (A, e, t) {
            for (var r = A.firstChild, n = void 0; r; r = n)
                if (((n = r.nextSibling), hn(r) && r.data.trim().length > 0)) e.textNodes.push(new Pr(r, e.styles));
                else if (gn(r)) {
                    var s = ln(r);
                    s.styles.isVisible() && (un(r, s, t) ? (s.flags |= 4) : wn(s.styles) && (s.flags |= 2), -1 !== an.indexOf(r.tagName) && (s.flags |= 8), e.elements.push(s), In(r) || Fn(r) || Tn(r) || cn(r, s, t));
                }
        },
        ln = function (A) {
            return pn(A) ? new Jr(A) : Hn(A) ? new Gr(A) : Fn(A) ? new kr(A) : Cn(A) ? new Wr(A) : dn(A) ? new Yr(A) : En(A) ? new rn(A) : Tn(A) ? new nn(A) : In(A) ? new sn(A) : Nn(A) ? new Bn(A) : new br(A);
        },
        Qn = function (A) {
            var e = ln(A);
            return (e.flags |= 4), cn(A, e, e), e;
        },
        un = function (A, e, t) {
            return e.styles.isPositionedWithZIndex() || e.styles.opacity < 1 || e.styles.isTransformed() || (fn(A) && t.styles.isTransparent());
        },
        wn = function (A) {
            return A.isPositioned() || A.isFloating();
        },
        hn = function (A) {
            return A.nodeType === Node.TEXT_NODE;
        },
        gn = function (A) {
            return A.nodeType === Node.ELEMENT_NODE;
        },
        Un = function (A) {
            return void 0 !== A.style;
        },
        Cn = function (A) {
            return "LI" === A.tagName;
        },
        dn = function (A) {
            return "OL" === A.tagName;
        },
        En = function (A) {
            return "INPUT" === A.tagName;
        },
        Fn = function (A) {
            return "svg" === A.tagName;
        },
        fn = function (A) {
            return "BODY" === A.tagName;
        },
        Hn = function (A) {
            return "CANVAS" === A.tagName;
        },
        pn = function (A) {
            return "IMG" === A.tagName;
        },
        Nn = function (A) {
            return "IFRAME" === A.tagName;
        },
        Kn = function (A) {
            return "STYLE" === A.tagName;
        },
        mn = function (A) {
            return "SCRIPT" === A.tagName;
        },
        In = function (A) {
            return "TEXTAREA" === A.tagName;
        },
        Tn = function (A) {
            return "SELECT" === A.tagName;
        },
        vn = (function () {
            function A() {
                this.counters = {};
            }
            return (
                (A.prototype.getCounterValue = function (A) {
                    var e = this.counters[A];
                    return e && e.length ? e[e.length - 1] : 1;
                }),
                (A.prototype.getCounterValues = function (A) {
                    var e = this.counters[A];
                    return e || [];
                }),
                (A.prototype.pop = function (A) {
                    var e = this;
                    A.forEach(function (A) {
                        return e.counters[A].pop();
                    });
                }),
                (A.prototype.parse = function (A) {
                    var e = this,
                        t = A.counterIncrement,
                        r = A.counterReset;
                    null !== t &&
                        t.forEach(function (A) {
                            var t = e.counters[A.counter];
                            t && (t[Math.max(0, t.length - 1)] += A.increment);
                        });
                    var n = [];
                    return (
                        r.forEach(function (A) {
                            var t = e.counters[A.counter];
                            n.push(A.counter), t || (t = e.counters[A.counter] = []), t.push(A.reset);
                        }),
                        n
                    );
                }),
                A
            );
        })(),
        Ln = { integers: [1e3, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1], values: ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"] },
        Rn = {
            integers: [9e3, 8e3, 7e3, 6e3, 5e3, 4e3, 3e3, 2e3, 1e3, 900, 800, 700, 600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            values: ["??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??"],
        },
        yn = {
            integers: [1e4, 9e3, 8e3, 7e3, 6e3, 5e3, 4e3, 3e3, 2e3, 1e3, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 19, 18, 17, 16, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            values: ["????", "????", "????", "????", "????", "????", "????", "????", "????", "????", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??", "????", "????", "????", "????", "????", "??", "??", "??", "??", "??", "??", "??", "??", "??", "??"],
        },
        bn = {
            integers: [1e4, 9e3, 8e3, 7e3, 6e3, 5e3, 4e3, 3e3, 2e3, 1e3, 900, 800, 700, 600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            values: ["???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???"],
        },
        On = function (A, e, t, r, n, s) {
            return A < e || A > t
                ? xn(A, n, s.length > 0)
                : r.integers.reduce(function (e, t, n) {
                      for (; A >= t; ) (A -= t), (e += r.values[n]);
                      return e;
                  }, "") + s;
        },
        Sn = function (A, e, t, r) {
            var n = "";
            do {
                t || A--, (n = r(A) + n), (A /= e);
            } while (A * e >= e);
            return n;
        },
        Mn = function (A, e, t, r, n) {
            var s = t - e + 1;
            return (
                (A < 0 ? "-" : "") +
                (Sn(Math.abs(A), s, r, function (A) {
                    return u(Math.floor(A % s) + e);
                }) +
                    n)
            );
        },
        Dn = function (A, e, t) {
            void 0 === t && (t = ". ");
            var r = e.length;
            return (
                Sn(Math.abs(A), r, !1, function (A) {
                    return e[Math.floor(A % r)];
                }) + t
            );
        },
        _n = function (A, e, t, r, n, s) {
            if (A < -9999 || A > 9999) return xn(A, Rt.CJK_DECIMAL, n.length > 0);
            var i = Math.abs(A),
                o = n;
            if (0 === i) return e[0] + o;
            for (var B = 0; i > 0 && B <= 4; B++) {
                var a = i % 10;
                0 === a && Hr(s, 1) && "" !== o
                    ? (o = e[a] + o)
                    : a > 1 || (1 === a && 0 === B) || (1 === a && 1 === B && Hr(s, 2)) || (1 === a && 1 === B && Hr(s, 4) && A > 100) || (1 === a && B > 1 && Hr(s, 8))
                    ? (o = e[a] + (B > 0 ? t[B - 1] : "") + o)
                    : 1 === a && B > 0 && (o = t[B - 1] + o),
                    (i = Math.floor(i / 10));
            }
            return (A < 0 ? r : "") + o;
        },
        xn = function (A, e, t) {
            var r = t ? ". " : "",
                n = t ? "???" : "",
                s = t ? ", " : "",
                i = t ? " " : "";
            switch (e) {
                case Rt.DISC:
                    return "???" + i;
                case Rt.CIRCLE:
                    return "???" + i;
                case Rt.SQUARE:
                    return "???" + i;
                case Rt.DECIMAL_LEADING_ZERO:
                    var o = Mn(A, 48, 57, !0, r);
                    return o.length < 4 ? "0" + o : o;
                case Rt.CJK_DECIMAL:
                    return Dn(A, "??????????????????????????????", n);
                case Rt.LOWER_ROMAN:
                    return On(A, 1, 3999, Ln, Rt.DECIMAL, r).toLowerCase();
                case Rt.UPPER_ROMAN:
                    return On(A, 1, 3999, Ln, Rt.DECIMAL, r);
                case Rt.LOWER_GREEK:
                    return Mn(A, 945, 969, !1, r);
                case Rt.LOWER_ALPHA:
                    return Mn(A, 97, 122, !1, r);
                case Rt.UPPER_ALPHA:
                    return Mn(A, 65, 90, !1, r);
                case Rt.ARABIC_INDIC:
                    return Mn(A, 1632, 1641, !0, r);
                case Rt.ARMENIAN:
                case Rt.UPPER_ARMENIAN:
                    return On(A, 1, 9999, Rn, Rt.DECIMAL, r);
                case Rt.LOWER_ARMENIAN:
                    return On(A, 1, 9999, Rn, Rt.DECIMAL, r).toLowerCase();
                case Rt.BENGALI:
                    return Mn(A, 2534, 2543, !0, r);
                case Rt.CAMBODIAN:
                case Rt.KHMER:
                    return Mn(A, 6112, 6121, !0, r);
                case Rt.CJK_EARTHLY_BRANCH:
                    return Dn(A, "????????????????????????????????????", n);
                case Rt.CJK_HEAVENLY_STEM:
                    return Dn(A, "??????????????????????????????", n);
                case Rt.CJK_IDEOGRAPHIC:
                case Rt.TRAD_CHINESE_INFORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "???", n, 14);
                case Rt.TRAD_CHINESE_FORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "???", n, 15);
                case Rt.SIMP_CHINESE_INFORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "???", n, 14);
                case Rt.SIMP_CHINESE_FORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "???", n, 15);
                case Rt.JAPANESE_INFORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "????????????", n, 0);
                case Rt.JAPANESE_FORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "????????????", n, 7);
                case Rt.KOREAN_HANGUL_FORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "????????????", s, 7);
                case Rt.KOREAN_HANJA_INFORMAL:
                    return _n(A, "??????????????????????????????", "????????????", "????????????", s, 0);
                case Rt.KOREAN_HANJA_FORMAL:
                    return _n(A, "??????????????????????????????", "?????????", "????????????", s, 7);
                case Rt.DEVANAGARI:
                    return Mn(A, 2406, 2415, !0, r);
                case Rt.GEORGIAN:
                    return On(A, 1, 19999, bn, Rt.DECIMAL, r);
                case Rt.GUJARATI:
                    return Mn(A, 2790, 2799, !0, r);
                case Rt.GURMUKHI:
                    return Mn(A, 2662, 2671, !0, r);
                case Rt.HEBREW:
                    return On(A, 1, 10999, yn, Rt.DECIMAL, r);
                case Rt.HIRAGANA:
                    return Dn(A, "????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????");
                case Rt.HIRAGANA_IROHA:
                    return Dn(A, "?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????");
                case Rt.KANNADA:
                    return Mn(A, 3302, 3311, !0, r);
                case Rt.KATAKANA:
                    return Dn(A, "????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????", n);
                case Rt.KATAKANA_IROHA:
                    return Dn(A, "?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????", n);
                case Rt.LAO:
                    return Mn(A, 3792, 3801, !0, r);
                case Rt.MONGOLIAN:
                    return Mn(A, 6160, 6169, !0, r);
                case Rt.MYANMAR:
                    return Mn(A, 4160, 4169, !0, r);
                case Rt.ORIYA:
                    return Mn(A, 2918, 2927, !0, r);
                case Rt.PERSIAN:
                    return Mn(A, 1776, 1785, !0, r);
                case Rt.TAMIL:
                    return Mn(A, 3046, 3055, !0, r);
                case Rt.TELUGU:
                    return Mn(A, 3174, 3183, !0, r);
                case Rt.THAI:
                    return Mn(A, 3664, 3673, !0, r);
                case Rt.TIBETAN:
                    return Mn(A, 3872, 3881, !0, r);
                case Rt.DECIMAL:
                default:
                    return Mn(A, 48, 57, !0, r);
            }
        },
        Pn = (function () {
            function A(A, e) {
                if (((this.options = e), (this.scrolledElements = []), (this.referenceElement = A), (this.counters = new vn()), (this.quoteDepth = 0), !A.ownerDocument)) throw new Error("Cloned element does not have an owner document");
                this.documentElement = this.cloneNode(A.ownerDocument.documentElement);
            }
            return (
                (A.prototype.toIFrame = function (A, e) {
                    var t = this,
                        r = zn(A, e);
                    if (!r.contentWindow) return Promise.reject("Unable to find iframe window");
                    var n = A.defaultView.pageXOffset,
                        s = A.defaultView.pageYOffset,
                        i = r.contentWindow,
                        o = i.document,
                        B = Xn(r).then(function () {
                            t.scrolledElements.forEach(Wn),
                                i &&
                                    (i.scrollTo(e.left, e.top),
                                    !/(iPad|iPhone|iPod)/g.test(navigator.userAgent) ||
                                        (i.scrollY === e.top && i.scrollX === e.left) ||
                                        ((o.documentElement.style.top = -e.top + "px"), (o.documentElement.style.left = -e.left + "px"), (o.documentElement.style.position = "absolute")));
                            var A = t.options.onclone;
                            return void 0 === t.clonedReferenceElement
                                ? Promise.reject("Error finding the " + t.referenceElement.nodeName + " in the cloned document")
                                : "function" == typeof A
                                ? Promise.resolve()
                                      .then(function () {
                                          return A(o);
                                      })
                                      .then(function () {
                                          return r;
                                      })
                                : r;
                        });
                    return o.open(), o.write(Gn(document.doctype) + "<html></html>"), kn(this.referenceElement.ownerDocument, n, s), o.replaceChild(o.adoptNode(this.documentElement), o.documentElement), o.close(), B;
                }),
                (A.prototype.createElementClone = function (A) {
                    return Hn(A) ? this.createCanvasClone(A) : Kn(A) ? this.createStyleClone(A) : A.cloneNode(!1);
                }),
                (A.prototype.createStyleClone = function (A) {
                    try {
                        var e = A.sheet;
                        if (e && e.cssRules) {
                            var t = [].slice.call(e.cssRules, 0).reduce(function (A, e) {
                                    return e && "string" == typeof e.cssText ? A + e.cssText : A;
                                }, ""),
                                r = A.cloneNode(!1);
                            return (r.textContent = t), r;
                        }
                    } catch (A) {
                        if ((Te.getInstance(this.options.id).error("Unable to access cssRules property", A), "SecurityError" !== A.name)) throw A;
                    }
                    return A.cloneNode(!1);
                }),
                (A.prototype.createCanvasClone = function (A) {
                    if (this.options.inlineImages && A.ownerDocument) {
                        var e = A.ownerDocument.createElement("img");
                        try {
                            return (e.src = A.toDataURL()), e;
                        } catch (A) {
                            Te.getInstance(this.options.id).info("Unable to clone canvas contents, canvas is tainted");
                        }
                    }
                    var t = A.cloneNode(!1);
                    try {
                        (t.width = A.width), (t.height = A.height);
                        var r = A.getContext("2d"),
                            n = t.getContext("2d");
                        return n && (r ? n.putImageData(r.getImageData(0, 0, A.width, A.height), 0, 0) : n.drawImage(A, 0, 0)), t;
                    } catch (A) {}
                    return t;
                }),
                (A.prototype.cloneNode = function (A) {
                    if (hn(A)) return document.createTextNode(A.data);
                    if (!A.ownerDocument) return A.cloneNode(!1);
                    var e = A.ownerDocument.defaultView;
                    if (Un(A) && e) {
                        var t = this.createElementClone(A),
                            r = e.getComputedStyle(A),
                            n = e.getComputedStyle(A, ":before"),
                            s = e.getComputedStyle(A, ":after");
                        this.referenceElement === A && (this.clonedReferenceElement = t), fn(t) && jn(t);
                        for (var i = this.counters.parse(new Rr(r)), o = this.resolvePseudoContent(A, t, n, Fr.BEFORE), B = A.firstChild; B; B = B.nextSibling)
                            (gn(B) && (mn(B) || B.hasAttribute("data-html2canvas-ignore") || ("function" == typeof this.options.ignoreElements && this.options.ignoreElements(B)))) ||
                                (this.options.copyStyles && gn(B) && Kn(B)) ||
                                t.appendChild(this.cloneNode(B));
                        o && t.insertBefore(o, t.firstChild);
                        var a = this.resolvePseudoContent(A, t, s, Fr.AFTER);
                        return (
                            a && t.appendChild(a),
                            this.counters.pop(i),
                            r && this.options.copyStyles && !Nn(A) && Jn(r, t),
                            (0 === A.scrollTop && 0 === A.scrollLeft) || this.scrolledElements.push([t, A.scrollLeft, A.scrollTop]),
                            (In(A) || Tn(A)) && (In(t) || Tn(t)) && (t.value = A.value),
                            t
                        );
                    }
                    return A.cloneNode(!1);
                }),
                (A.prototype.resolvePseudoContent = function (A, e, t, r) {
                    var n = this;
                    if (t) {
                        var s = t.content,
                            i = e.ownerDocument;
                        if (i && s && "none" !== s && "-moz-alt-content" !== s && "none" !== t.display) {
                            this.counters.parse(new Rr(t));
                            var o = new Lr(t),
                                B = i.createElement("html2canvaspseudoelement");
                            return (
                                Jn(t, B),
                                o.content.forEach(function (e) {
                                    if (e.type === U.STRING_TOKEN) B.appendChild(i.createTextNode(e.value));
                                    else if (e.type === U.URL_TOKEN) {
                                        var t = i.createElement("img");
                                        (t.src = e.value), (t.style.opacity = "1"), B.appendChild(t);
                                    } else if (e.type === U.FUNCTION) {
                                        if ("attr" === e.name) {
                                            var r = e.values.filter(_A);
                                            r.length && B.appendChild(i.createTextNode(A.getAttribute(r[0].value) || ""));
                                        } else if ("counter" === e.name) {
                                            var s = e.values.filter(zA),
                                                a = s[0],
                                                c = s[1];
                                            if (a && _A(a)) {
                                                var l = n.counters.getCounterValue(a.value),
                                                    Q = c && _A(c) ? Ot.parse(c.value) : Rt.DECIMAL;
                                                B.appendChild(i.createTextNode(xn(l, Q, !1)));
                                            }
                                        } else if ("counters" === e.name) {
                                            var u = e.values.filter(zA),
                                                w = ((a = u[0]), u[1]);
                                            c = u[2];
                                            if (a && _A(a)) {
                                                var h = n.counters.getCounterValues(a.value),
                                                    g = c && _A(c) ? Ot.parse(c.value) : Rt.DECIMAL,
                                                    C = w && w.type === U.STRING_TOKEN ? w.value : "",
                                                    d = h
                                                        .map(function (A) {
                                                            return xn(A, g, !1);
                                                        })
                                                        .join(C);
                                                B.appendChild(i.createTextNode(d));
                                            }
                                        }
                                    } else if (e.type === U.IDENT_TOKEN)
                                        switch (e.value) {
                                            case "open-quote":
                                                B.appendChild(i.createTextNode(Ir(o.quotes, n.quoteDepth++, !0)));
                                                break;
                                            case "close-quote":
                                                B.appendChild(i.createTextNode(Ir(o.quotes, --n.quoteDepth, !1)));
                                        }
                                }),
                                (B.className = Yn + " " + qn),
                                (e.className += r === Fr.BEFORE ? " " + Yn : " " + qn),
                                B
                            );
                        }
                    }
                }),
                A
            );
        })();
    !(function (A) {
        (A[(A.BEFORE = 0)] = "BEFORE"), (A[(A.AFTER = 1)] = "AFTER");
    })(Fr || (Fr = {}));
    var Vn,
        zn = function (A, e) {
            var t = A.createElement("iframe");
            return (
                (t.className = "html2canvas-container"),
                (t.style.visibility = "hidden"),
                (t.style.position = "fixed"),
                (t.style.left = "-10000px"),
                (t.style.top = "0px"),
                (t.style.border = "0"),
                (t.width = e.width.toString()),
                (t.height = e.height.toString()),
                (t.scrolling = "no"),
                t.setAttribute("data-html2canvas-ignore", "true"),
                A.body.appendChild(t),
                t
            );
        },
        Xn = function (A) {
            return new Promise(function (e, t) {
                var r = A.contentWindow;
                if (!r) return t("No window assigned for iframe");
                var n = r.document;
                r.onload = A.onload = n.onreadystatechange = function () {
                    r.onload = A.onload = n.onreadystatechange = null;
                    var t = setInterval(function () {
                        n.body.childNodes.length > 0 && "complete" === n.readyState && (clearInterval(t), e(A));
                    }, 50);
                };
            });
        },
        Jn = function (A, e) {
            for (var t = A.length - 1; t >= 0; t--) {
                var r = A.item(t);
                "content" !== r && e.style.setProperty(r, A.getPropertyValue(r));
            }
            return e;
        },
        Gn = function (A) {
            var e = "";
            return A && ((e += "<!DOCTYPE "), A.name && (e += A.name), A.internalSubset && (e += A.internalSubset), A.publicId && (e += '"' + A.publicId + '"'), A.systemId && (e += '"' + A.systemId + '"'), (e += ">")), e;
        },
        kn = function (A, e, t) {
            A && A.defaultView && (e !== A.defaultView.pageXOffset || t !== A.defaultView.pageYOffset) && A.defaultView.scrollTo(e, t);
        },
        Wn = function (A) {
            var e = A[0],
                t = A[1],
                r = A[2];
            (e.scrollLeft = t), (e.scrollTop = r);
        },
        Yn = "___html2canvas___pseudoelement_before",
        qn = "___html2canvas___pseudoelement_after",
        jn = function (A) {
            Zn(A, "." + Yn + ':before{\n    content: "" !important;\n    display: none !important;\n}\n         .' + qn + ':after{\n    content: "" !important;\n    display: none !important;\n}');
        },
        Zn = function (A, e) {
            var t = A.ownerDocument;
            if (t) {
                var r = t.createElement("style");
                (r.textContent = e), A.appendChild(r);
            }
        };
    !(function (A) {
        (A[(A.VECTOR = 0)] = "VECTOR"), (A[(A.BEZIER_CURVE = 1)] = "BEZIER_CURVE");
    })(Vn || (Vn = {}));
    var $n,
        As = function (A, e) {
            return (
                A.length === e.length &&
                A.some(function (A, t) {
                    return A === e[t];
                })
            );
        },
        es = (function () {
            function A(A, e) {
                (this.type = Vn.VECTOR), (this.x = A), (this.y = e);
            }
            return (
                (A.prototype.add = function (e, t) {
                    return new A(this.x + e, this.y + t);
                }),
                A
            );
        })(),
        ts = function (A, e, t) {
            return new es(A.x + (e.x - A.x) * t, A.y + (e.y - A.y) * t);
        },
        rs = (function () {
            function A(A, e, t, r) {
                (this.type = Vn.BEZIER_CURVE), (this.start = A), (this.startControl = e), (this.endControl = t), (this.end = r);
            }
            return (
                (A.prototype.subdivide = function (e, t) {
                    var r = ts(this.start, this.startControl, e),
                        n = ts(this.startControl, this.endControl, e),
                        s = ts(this.endControl, this.end, e),
                        i = ts(r, n, e),
                        o = ts(n, s, e),
                        B = ts(i, o, e);
                    return t ? new A(this.start, r, i, B) : new A(B, o, s, this.end);
                }),
                (A.prototype.add = function (e, t) {
                    return new A(this.start.add(e, t), this.startControl.add(e, t), this.endControl.add(e, t), this.end.add(e, t));
                }),
                (A.prototype.reverse = function () {
                    return new A(this.end, this.endControl, this.startControl, this.start);
                }),
                A
            );
        })(),
        ns = function (A) {
            return A.type === Vn.BEZIER_CURVE;
        },
        ss = (function () {
            return function (A) {
                var e = A.styles,
                    t = A.bounds,
                    r = ZA(e.borderTopLeftRadius, t.width, t.height),
                    n = r[0],
                    s = r[1],
                    i = ZA(e.borderTopRightRadius, t.width, t.height),
                    o = i[0],
                    B = i[1],
                    a = ZA(e.borderBottomRightRadius, t.width, t.height),
                    c = a[0],
                    l = a[1],
                    Q = ZA(e.borderBottomLeftRadius, t.width, t.height),
                    u = Q[0],
                    w = Q[1],
                    h = [];
                h.push((n + o) / t.width), h.push((u + c) / t.width), h.push((s + w) / t.height), h.push((B + l) / t.height);
                var g = Math.max.apply(Math, h);
                g > 1 && ((n /= g), (s /= g), (o /= g), (B /= g), (c /= g), (l /= g), (u /= g), (w /= g));
                var U = t.width - o,
                    C = t.height - l,
                    d = t.width - c,
                    E = t.height - w,
                    F = e.borderTopWidth,
                    f = e.borderRightWidth,
                    H = e.borderBottomWidth,
                    p = e.borderLeftWidth,
                    N = $A(e.paddingTop, A.bounds.width),
                    K = $A(e.paddingRight, A.bounds.width),
                    m = $A(e.paddingBottom, A.bounds.width),
                    I = $A(e.paddingLeft, A.bounds.width);
                (this.topLeftBorderBox = n > 0 || s > 0 ? is(t.left, t.top, n, s, $n.TOP_LEFT) : new es(t.left, t.top)),
                    (this.topRightBorderBox = o > 0 || B > 0 ? is(t.left + U, t.top, o, B, $n.TOP_RIGHT) : new es(t.left + t.width, t.top)),
                    (this.bottomRightBorderBox = c > 0 || l > 0 ? is(t.left + d, t.top + C, c, l, $n.BOTTOM_RIGHT) : new es(t.left + t.width, t.top + t.height)),
                    (this.bottomLeftBorderBox = u > 0 || w > 0 ? is(t.left, t.top + E, u, w, $n.BOTTOM_LEFT) : new es(t.left, t.top + t.height)),
                    (this.topLeftPaddingBox = n > 0 || s > 0 ? is(t.left + p, t.top + F, Math.max(0, n - p), Math.max(0, s - F), $n.TOP_LEFT) : new es(t.left + p, t.top + F)),
                    (this.topRightPaddingBox = o > 0 || B > 0 ? is(t.left + Math.min(U, t.width + p), t.top + F, U > t.width + p ? 0 : o - p, B - F, $n.TOP_RIGHT) : new es(t.left + t.width - f, t.top + F)),
                    (this.bottomRightPaddingBox = c > 0 || l > 0 ? is(t.left + Math.min(d, t.width - p), t.top + Math.min(C, t.height + F), Math.max(0, c - f), l - H, $n.BOTTOM_RIGHT) : new es(t.left + t.width - f, t.top + t.height - H)),
                    (this.bottomLeftPaddingBox = u > 0 || w > 0 ? is(t.left + p, t.top + E, Math.max(0, u - p), w - H, $n.BOTTOM_LEFT) : new es(t.left + p, t.top + t.height - H)),
                    (this.topLeftContentBox = n > 0 || s > 0 ? is(t.left + p + I, t.top + F + N, Math.max(0, n - (p + I)), Math.max(0, s - (F + N)), $n.TOP_LEFT) : new es(t.left + p + I, t.top + F + N)),
                    (this.topRightContentBox = o > 0 || B > 0 ? is(t.left + Math.min(U, t.width + p + I), t.top + F + N, U > t.width + p + I ? 0 : o - p + I, B - (F + N), $n.TOP_RIGHT) : new es(t.left + t.width - (f + K), t.top + F + N)),
                    (this.bottomRightContentBox =
                        c > 0 || l > 0
                            ? is(t.left + Math.min(d, t.width - (p + I)), t.top + Math.min(C, t.height + F + N), Math.max(0, c - (f + K)), l - (H + m), $n.BOTTOM_RIGHT)
                            : new es(t.left + t.width - (f + K), t.top + t.height - (H + m))),
                    (this.bottomLeftContentBox = u > 0 || w > 0 ? is(t.left + p + I, t.top + E, Math.max(0, u - (p + I)), w - (H + m), $n.BOTTOM_LEFT) : new es(t.left + p + I, t.top + t.height - (H + m)));
            };
        })();
    !(function (A) {
        (A[(A.TOP_LEFT = 0)] = "TOP_LEFT"), (A[(A.TOP_RIGHT = 1)] = "TOP_RIGHT"), (A[(A.BOTTOM_RIGHT = 2)] = "BOTTOM_RIGHT"), (A[(A.BOTTOM_LEFT = 3)] = "BOTTOM_LEFT");
    })($n || ($n = {}));
    var is = function (A, e, t, r, n) {
            var s = ((Math.sqrt(2) - 1) / 3) * 4,
                i = t * s,
                o = r * s,
                B = A + t,
                a = e + r;
            switch (n) {
                case $n.TOP_LEFT:
                    return new rs(new es(A, a), new es(A, a - o), new es(B - i, e), new es(B, e));
                case $n.TOP_RIGHT:
                    return new rs(new es(A, e), new es(A + i, e), new es(B, a - o), new es(B, a));
                case $n.BOTTOM_RIGHT:
                    return new rs(new es(B, e), new es(B, e + o), new es(A + i, a), new es(A, a));
                case $n.BOTTOM_LEFT:
                default:
                    return new rs(new es(B, a), new es(B - i, a), new es(A, e + o), new es(A, e));
            }
        },
        os = function (A) {
            return [A.topLeftBorderBox, A.topRightBorderBox, A.bottomRightBorderBox, A.bottomLeftBorderBox];
        },
        Bs = function (A) {
            return [A.topLeftPaddingBox, A.topRightPaddingBox, A.bottomRightPaddingBox, A.bottomLeftPaddingBox];
        },
        as = (function () {
            return function (A, e, t) {
                (this.type = 0), (this.offsetX = A), (this.offsetY = e), (this.matrix = t), (this.target = 6);
            };
        })(),
        cs = (function () {
            return function (A, e) {
                (this.type = 1), (this.target = e), (this.path = A);
            };
        })(),
        ls = (function () {
            return function (A) {
                (this.element = A),
                    (this.inlineLevel = []),
                    (this.nonInlineLevel = []),
                    (this.negativeZIndex = []),
                    (this.zeroOrAutoZIndexOrTransformedOrOpacity = []),
                    (this.positiveZIndex = []),
                    (this.nonPositionedFloats = []),
                    (this.nonPositionedInlineLevel = []);
            };
        })(),
        Qs = (function () {
            function A(A, e) {
                if (((this.container = A), (this.effects = e.slice(0)), (this.curves = new ss(A)), null !== A.styles.transform)) {
                    var t = A.bounds.left + A.styles.transformOrigin[0].number,
                        r = A.bounds.top + A.styles.transformOrigin[1].number,
                        n = A.styles.transform;
                    this.effects.push(new as(t, r, n));
                }
                if (A.styles.overflowX !== bt.VISIBLE) {
                    var s = os(this.curves),
                        i = Bs(this.curves);
                    As(s, i) ? this.effects.push(new cs(s, 6)) : (this.effects.push(new cs(s, 2)), this.effects.push(new cs(i, 4)));
                }
            }
            return (
                (A.prototype.getParentEffects = function () {
                    var A = this.effects.slice(0);
                    if (this.container.styles.overflowX !== bt.VISIBLE) {
                        var e = os(this.curves),
                            t = Bs(this.curves);
                        As(e, t) || A.push(new cs(t, 6));
                    }
                    return A;
                }),
                A
            );
        })(),
        us = function (A, e, t, r) {
            A.container.elements.forEach(function (n) {
                var s = Hr(n.flags, 4),
                    i = Hr(n.flags, 2),
                    o = new Qs(n, A.getParentEffects());
                Hr(n.styles.display, 2048) && r.push(o);
                var B = Hr(n.flags, 8) ? [] : r;
                if (s || i) {
                    var a = s || n.styles.isPositioned() ? t : e,
                        c = new ls(o);
                    if (n.styles.isPositioned() || n.styles.opacity < 1 || n.styles.isTransformed()) {
                        var l = n.styles.zIndex.order;
                        if (l < 0) {
                            var Q = 0;
                            a.negativeZIndex.some(function (A, e) {
                                return l > A.element.container.styles.zIndex.order && ((Q = e), !0);
                            }),
                                a.negativeZIndex.splice(Q, 0, c);
                        } else if (l > 0) {
                            var u = 0;
                            a.positiveZIndex.some(function (A, e) {
                                return l > A.element.container.styles.zIndex.order && ((u = e + 1), !0);
                            }),
                                a.positiveZIndex.splice(u, 0, c);
                        } else a.zeroOrAutoZIndexOrTransformedOrOpacity.push(c);
                    } else n.styles.isFloating() ? a.nonPositionedFloats.push(c) : a.nonPositionedInlineLevel.push(c);
                    us(o, c, s ? c : t, B);
                } else n.styles.isInlineLevel() ? e.inlineLevel.push(o) : e.nonInlineLevel.push(o), us(o, e, t, B);
                Hr(n.flags, 8) && ws(n, B);
            });
        },
        ws = function (A, e) {
            for (var t = A instanceof Yr ? A.start : 1, r = A instanceof Yr && A.reversed, n = 0; n < e.length; n++) {
                var s = e[n];
                s.container instanceof Wr && "number" == typeof s.container.value && 0 !== s.container.value && (t = s.container.value), (s.listValue = xn(t, s.container.styles.listStyleType, !0)), (t += r ? -1 : 1);
            }
        },
        hs = function (A, e, t, r) {
            var n = [];
            return (
                ns(A) ? n.push(A.subdivide(0.5, !1)) : n.push(A), ns(t) ? n.push(t.subdivide(0.5, !0)) : n.push(t), ns(r) ? n.push(r.subdivide(0.5, !0).reverse()) : n.push(r), ns(e) ? n.push(e.subdivide(0.5, !1).reverse()) : n.push(e), n
            );
        },
        gs = function (A) {
            var e = A.bounds,
                t = A.styles;
            return e.add(t.borderLeftWidth, t.borderTopWidth, -(t.borderRightWidth + t.borderLeftWidth), -(t.borderTopWidth + t.borderBottomWidth));
        },
        Us = function (A) {
            var e = A.styles,
                t = A.bounds,
                r = $A(e.paddingLeft, t.width),
                n = $A(e.paddingRight, t.width),
                s = $A(e.paddingTop, t.width),
                i = $A(e.paddingBottom, t.width);
            return t.add(r + e.borderLeftWidth, s + e.borderTopWidth, -(e.borderRightWidth + e.borderLeftWidth + r + n), -(e.borderTopWidth + e.borderBottomWidth + s + i));
        },
        Cs = function (A, e, t) {
            var r = (function (A, e) {
                    return 0 === A ? e.bounds : 2 === A ? Us(e) : gs(e);
                })(fs(A.styles.backgroundOrigin, e), A),
                n = (function (A, e) {
                    return A === Qe.BORDER_BOX ? e.bounds : A === Qe.CONTENT_BOX ? Us(e) : gs(e);
                })(fs(A.styles.backgroundClip, e), A),
                s = Fs(fs(A.styles.backgroundSize, e), t, r),
                i = s[0],
                o = s[1],
                B = ZA(fs(A.styles.backgroundPosition, e), r.width - i, r.height - o);
            return [Hs(fs(A.styles.backgroundRepeat, e), B, s, r, n), Math.round(r.left + B[0]), Math.round(r.top + B[1]), i, o];
        },
        ds = function (A) {
            return _A(A) && A.value === Ye.AUTO;
        },
        Es = function (A) {
            return "number" == typeof A;
        },
        Fs = function (A, e, t) {
            var r = e[0],
                n = e[1],
                s = e[2],
                i = A[0],
                o = A[1];
            if (kA(i) && o && kA(o)) return [$A(i, t.width), $A(o, t.height)];
            var B = Es(s);
            if (_A(i) && (i.value === Ye.CONTAIN || i.value === Ye.COVER)) return Es(s) ? (t.width / t.height < s != (i.value === Ye.COVER) ? [t.width, t.width / s] : [t.height * s, t.height]) : [t.width, t.height];
            var a = Es(r),
                c = Es(n),
                l = a || c;
            if (ds(i) && (!o || ds(o))) return a && c ? [r, n] : B || l ? (l && B ? [a ? r : n * s, c ? n : r / s] : [a ? r : t.width, c ? n : t.height]) : [t.width, t.height];
            if (B) {
                var Q = 0,
                    u = 0;
                return kA(i) ? (Q = $A(i, t.width)) : kA(o) && (u = $A(o, t.height)), ds(i) ? (Q = u * s) : (o && !ds(o)) || (u = Q / s), [Q, u];
            }
            var w = null,
                h = null;
            if (
                (kA(i) ? (w = $A(i, t.width)) : o && kA(o) && (h = $A(o, t.height)),
                null === w || (o && !ds(o)) || (h = a && c ? (w / r) * n : t.height),
                null !== h && ds(i) && (w = a && c ? (h / n) * r : t.width),
                null !== w && null !== h)
            )
                return [w, h];
            throw new Error("Unable to calculate background-size for element");
        },
        fs = function (A, e) {
            var t = A[e];
            return void 0 === t ? A[0] : t;
        },
        Hs = function (A, e, t, r, n) {
            var s = e[0],
                i = e[1],
                o = t[0],
                B = t[1];
            switch (A) {
                case ze.REPEAT_X:
                    return [
                        new es(Math.round(r.left), Math.round(r.top + i)),
                        new es(Math.round(r.left + r.width), Math.round(r.top + i)),
                        new es(Math.round(r.left + r.width), Math.round(B + r.top + i)),
                        new es(Math.round(r.left), Math.round(B + r.top + i)),
                    ];
                case ze.REPEAT_Y:
                    return [
                        new es(Math.round(r.left + s), Math.round(r.top)),
                        new es(Math.round(r.left + s + o), Math.round(r.top)),
                        new es(Math.round(r.left + s + o), Math.round(r.height + r.top)),
                        new es(Math.round(r.left + s), Math.round(r.height + r.top)),
                    ];
                case ze.NO_REPEAT:
                    return [
                        new es(Math.round(r.left + s), Math.round(r.top + i)),
                        new es(Math.round(r.left + s + o), Math.round(r.top + i)),
                        new es(Math.round(r.left + s + o), Math.round(r.top + i + B)),
                        new es(Math.round(r.left + s), Math.round(r.top + i + B)),
                    ];
                default:
                    return [
                        new es(Math.round(n.left), Math.round(n.top)),
                        new es(Math.round(n.left + n.width), Math.round(n.top)),
                        new es(Math.round(n.left + n.width), Math.round(n.height + n.top)),
                        new es(Math.round(n.left), Math.round(n.height + n.top)),
                    ];
            }
        },
        ps = (function () {
            function A(A) {
                (this._data = {}), (this._document = A);
            }
            return (
                (A.prototype.parseMetrics = function (A, e) {
                    var t = this._document.createElement("div"),
                        r = this._document.createElement("img"),
                        n = this._document.createElement("span"),
                        s = this._document.body;
                    (t.style.visibility = "hidden"),
                        (t.style.fontFamily = A),
                        (t.style.fontSize = e),
                        (t.style.margin = "0"),
                        (t.style.padding = "0"),
                        s.appendChild(t),
                        (r.src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"),
                        (r.width = 1),
                        (r.height = 1),
                        (r.style.margin = "0"),
                        (r.style.padding = "0"),
                        (r.style.verticalAlign = "baseline"),
                        (n.style.fontFamily = A),
                        (n.style.fontSize = e),
                        (n.style.margin = "0"),
                        (n.style.padding = "0"),
                        n.appendChild(this._document.createTextNode("Hidden Text")),
                        t.appendChild(n),
                        t.appendChild(r);
                    var i = r.offsetTop - n.offsetTop + 2;
                    t.removeChild(n), t.appendChild(this._document.createTextNode("Hidden Text")), (t.style.lineHeight = "normal"), (r.style.verticalAlign = "super");
                    var o = r.offsetTop - t.offsetTop + 2;
                    return s.removeChild(t), { baseline: i, middle: o };
                }),
                (A.prototype.getMetrics = function (A, e) {
                    var t = A + " " + e;
                    return void 0 === this._data[t] && (this._data[t] = this.parseMetrics(A, e)), this._data[t];
                }),
                A
            );
        })(),
        Ns = (function () {
            function A(A) {
                (this._activeEffects = []),
                    (this.canvas = A.canvas ? A.canvas : document.createElement("canvas")),
                    (this.ctx = this.canvas.getContext("2d")),
                    (this.options = A),
                    (this.canvas.width = Math.floor(A.width * A.scale)),
                    (this.canvas.height = Math.floor(A.height * A.scale)),
                    (this.canvas.style.width = A.width + "px"),
                    (this.canvas.style.height = A.height + "px"),
                    (this.fontMetrics = new ps(document)),
                    this.ctx.scale(this.options.scale, this.options.scale),
                    this.ctx.translate(-A.x + A.scrollX, -A.y + A.scrollY),
                    (this.ctx.textBaseline = "bottom"),
                    (this._activeEffects = []),
                    Te.getInstance(A.id).debug("Canvas renderer initialized (" + A.width + "x" + A.height + " at " + A.x + "," + A.y + ") with scale " + A.scale);
            }
            return (
                (A.prototype.applyEffects = function (A, e) {
                    for (var t = this; this._activeEffects.length; ) this.popEffect();
                    A.filter(function (A) {
                        return Hr(A.target, e);
                    }).forEach(function (A) {
                        return t.applyEffect(A);
                    });
                }),
                (A.prototype.applyEffect = function (A) {
                    this.ctx.save(),
                        (function (A) {
                            return 0 === A.type;
                        })(A) && (this.ctx.translate(A.offsetX, A.offsetY), this.ctx.transform(A.matrix[0], A.matrix[1], A.matrix[2], A.matrix[3], A.matrix[4], A.matrix[5]), this.ctx.translate(-A.offsetX, -A.offsetY)),
                        (function (A) {
                            return 1 === A.type;
                        })(A) && (this.path(A.path), this.ctx.clip()),
                        this._activeEffects.push(A);
                }),
                (A.prototype.popEffect = function () {
                    this._activeEffects.pop(), this.ctx.restore();
                }),
                (A.prototype.renderStack = function (A) {
                    return B(this, void 0, void 0, function () {
                        var e;
                        return a(this, function (t) {
                            switch (t.label) {
                                case 0:
                                    return (e = A.element.container.styles).isVisible() ? ((this.ctx.globalAlpha = e.opacity), [4, this.renderStackContent(A)]) : [3, 2];
                                case 1:
                                    t.sent(), (t.label = 2);
                                case 2:
                                    return [2];
                            }
                        });
                    });
                }),
                (A.prototype.renderNode = function (A) {
                    return B(this, void 0, void 0, function () {
                        return a(this, function (e) {
                            switch (e.label) {
                                case 0:
                                    return A.container.styles.isVisible() ? [4, this.renderNodeBackgroundAndBorders(A)] : [3, 3];
                                case 1:
                                    return e.sent(), [4, this.renderNodeContent(A)];
                                case 2:
                                    e.sent(), (e.label = 3);
                                case 3:
                                    return [2];
                            }
                        });
                    });
                }),
                (A.prototype.renderTextWithLetterSpacing = function (A, e) {
                    var t = this;
                    0 === e
                        ? this.ctx.fillText(A.text, A.bounds.left, A.bounds.top + A.bounds.height)
                        : Q(A.text)
                              .map(function (A) {
                                  return u(A);
                              })
                              .reduce(function (e, r) {
                                  return t.ctx.fillText(r, e, A.bounds.top + A.bounds.height), e + t.ctx.measureText(r).width;
                              }, A.bounds.left);
                }),
                (A.prototype.createFontStyle = function (A) {
                    var e = A.fontVariant
                            .filter(function (A) {
                                return "normal" === A || "small-caps" === A;
                            })
                            .join(""),
                        t = A.fontFamily.join(", "),
                        r = MA(A.fontSize) ? "" + A.fontSize.number + A.fontSize.unit : A.fontSize.number + "px";
                    return [[A.fontStyle, e, A.fontWeight, r, t].join(" "), t, r];
                }),
                (A.prototype.renderTextNode = function (A, e) {
                    return B(this, void 0, void 0, function () {
                        var t,
                            r,
                            n,
                            s,
                            i = this;
                        return a(this, function (o) {
                            return (
                                (t = this.createFontStyle(e)),
                                (r = t[0]),
                                (n = t[1]),
                                (s = t[2]),
                                (this.ctx.font = r),
                                A.textBounds.forEach(function (A) {
                                    (i.ctx.fillStyle = ie(e.color)), i.renderTextWithLetterSpacing(A, e.letterSpacing);
                                    var t = e.textShadow;
                                    t.length &&
                                        A.text.trim().length &&
                                        (t
                                            .slice(0)
                                            .reverse()
                                            .forEach(function (e) {
                                                (i.ctx.shadowColor = ie(e.color)),
                                                    (i.ctx.shadowOffsetX = e.offsetX.number * i.options.scale),
                                                    (i.ctx.shadowOffsetY = e.offsetY.number * i.options.scale),
                                                    (i.ctx.shadowBlur = e.blur.number),
                                                    i.ctx.fillText(A.text, A.bounds.left, A.bounds.top + A.bounds.height);
                                            }),
                                        (i.ctx.shadowColor = ""),
                                        (i.ctx.shadowOffsetX = 0),
                                        (i.ctx.shadowOffsetY = 0),
                                        (i.ctx.shadowBlur = 0)),
                                        e.textDecorationLine.length &&
                                            ((i.ctx.fillStyle = ie(e.textDecorationColor || e.color)),
                                            e.textDecorationLine.forEach(function (e) {
                                                switch (e) {
                                                    case 1:
                                                        var t = i.fontMetrics.getMetrics(n, s).baseline;
                                                        i.ctx.fillRect(A.bounds.left, Math.round(A.bounds.top + t), A.bounds.width, 1);
                                                        break;
                                                    case 2:
                                                        i.ctx.fillRect(A.bounds.left, Math.round(A.bounds.top), A.bounds.width, 1);
                                                        break;
                                                    case 3:
                                                        var r = i.fontMetrics.getMetrics(n, s).middle;
                                                        i.ctx.fillRect(A.bounds.left, Math.ceil(A.bounds.top + r), A.bounds.width, 1);
                                                }
                                            }));
                                }),
                                [2]
                            );
                        });
                    });
                }),
                (A.prototype.renderReplacedElement = function (A, e, t) {
                    if (t && A.intrinsicWidth > 0 && A.intrinsicHeight > 0) {
                        var r = Us(A),
                            n = Bs(e);
                        this.path(n), this.ctx.save(), this.ctx.clip(), this.ctx.drawImage(t, 0, 0, A.intrinsicWidth, A.intrinsicHeight, r.left, r.top, r.width, r.height), this.ctx.restore();
                    }
                }),
                (A.prototype.renderNodeContent = function (e) {
                    return B(this, void 0, void 0, function () {
                        var t, r, n, s, i, o, B, l, Q, u, w, h, g, C;
                        return a(this, function (a) {
                            switch (a.label) {
                                case 0:
                                    this.applyEffects(e.effects, 4), (t = e.container), (r = e.curves), (n = t.styles), (s = 0), (i = t.textNodes), (a.label = 1);
                                case 1:
                                    return s < i.length ? ((o = i[s]), [4, this.renderTextNode(o, n)]) : [3, 4];
                                case 2:
                                    a.sent(), (a.label = 3);
                                case 3:
                                    return s++, [3, 1];
                                case 4:
                                    if (!(t instanceof Jr)) return [3, 8];
                                    a.label = 5;
                                case 5:
                                    return a.trys.push([5, 7, , 8]), [4, this.options.cache.match(t.src)];
                                case 6:
                                    return (h = a.sent()), this.renderReplacedElement(t, r, h), [3, 8];
                                case 7:
                                    return a.sent(), Te.getInstance(this.options.id).error("Error loading image " + t.src), [3, 8];
                                case 8:
                                    if ((t instanceof Gr && this.renderReplacedElement(t, r, t.canvas), !(t instanceof kr))) return [3, 12];
                                    a.label = 9;
                                case 9:
                                    return a.trys.push([9, 11, , 12]), [4, this.options.cache.match(t.svg)];
                                case 10:
                                    return (h = a.sent()), this.renderReplacedElement(t, r, h), [3, 12];
                                case 11:
                                    return a.sent(), Te.getInstance(this.options.id).error("Error loading svg " + t.svg.substring(0, 255)), [3, 12];
                                case 12:
                                    return t instanceof Bn && t.tree
                                        ? [
                                              4,
                                              new A({
                                                  id: this.options.id,
                                                  scale: this.options.scale,
                                                  backgroundColor: t.backgroundColor,
                                                  x: 0,
                                                  y: 0,
                                                  scrollX: 0,
                                                  scrollY: 0,
                                                  width: t.width,
                                                  height: t.height,
                                                  cache: this.options.cache,
                                                  windowWidth: t.width,
                                                  windowHeight: t.height,
                                              }).render(t.tree),
                                          ]
                                        : [3, 14];
                                case 13:
                                    (B = a.sent()), this.ctx.drawImage(B, 0, 0, t.width, t.width, t.bounds.left, t.bounds.top, t.bounds.width, t.bounds.height), (a.label = 14);
                                case 14:
                                    if (
                                        (t instanceof rn &&
                                            ((l = Math.min(t.bounds.width, t.bounds.height)),
                                            t.type === An
                                                ? t.checked &&
                                                  (this.ctx.save(),
                                                  this.path([
                                                      new es(t.bounds.left + 0.39363 * l, t.bounds.top + 0.79 * l),
                                                      new es(t.bounds.left + 0.16 * l, t.bounds.top + 0.5549 * l),
                                                      new es(t.bounds.left + 0.27347 * l, t.bounds.top + 0.44071 * l),
                                                      new es(t.bounds.left + 0.39694 * l, t.bounds.top + 0.5649 * l),
                                                      new es(t.bounds.left + 0.72983 * l, t.bounds.top + 0.23 * l),
                                                      new es(t.bounds.left + 0.84 * l, t.bounds.top + 0.34085 * l),
                                                      new es(t.bounds.left + 0.39363 * l, t.bounds.top + 0.79 * l),
                                                  ]),
                                                  (this.ctx.fillStyle = ie(707406591)),
                                                  this.ctx.fill(),
                                                  this.ctx.restore())
                                                : t.type === en &&
                                                  t.checked &&
                                                  (this.ctx.save(),
                                                  this.ctx.beginPath(),
                                                  this.ctx.arc(t.bounds.left + l / 2, t.bounds.top + l / 2, l / 4, 0, 2 * Math.PI, !0),
                                                  (this.ctx.fillStyle = ie(707406591)),
                                                  this.ctx.fill(),
                                                  this.ctx.restore())),
                                        Ks(t) && t.value.length)
                                    ) {
                                        switch (
                                            ((this.ctx.font = this.createFontStyle(n)[0]),
                                            (this.ctx.fillStyle = ie(n.color)),
                                            (this.ctx.textBaseline = "middle"),
                                            (this.ctx.textAlign = Is(t.styles.textAlign)),
                                            (C = Us(t)),
                                            (Q = 0),
                                            t.styles.textAlign)
                                        ) {
                                            case zt.CENTER:
                                                Q += C.width / 2;
                                                break;
                                            case zt.RIGHT:
                                                Q += C.width;
                                        }
                                        (u = C.add(Q, 0, 0, -C.height / 2 + 1)),
                                            this.ctx.save(),
                                            this.path([new es(C.left, C.top), new es(C.left + C.width, C.top), new es(C.left + C.width, C.top + C.height), new es(C.left, C.top + C.height)]),
                                            this.ctx.clip(),
                                            this.renderTextWithLetterSpacing(new Or(t.value, u), n.letterSpacing),
                                            this.ctx.restore(),
                                            (this.ctx.textBaseline = "bottom"),
                                            (this.ctx.textAlign = "left");
                                    }
                                    if (!Hr(t.styles.display, 2048)) return [3, 20];
                                    if (null === t.styles.listStyleImage) return [3, 19];
                                    if ((w = t.styles.listStyleImage).type !== ge.URL) return [3, 18];
                                    (h = void 0), (g = w.url), (a.label = 15);
                                case 15:
                                    return a.trys.push([15, 17, , 18]), [4, this.options.cache.match(g)];
                                case 16:
                                    return (h = a.sent()), this.ctx.drawImage(h, t.bounds.left - (h.width + 10), t.bounds.top), [3, 18];
                                case 17:
                                    return a.sent(), Te.getInstance(this.options.id).error("Error loading list-style-image " + g), [3, 18];
                                case 18:
                                    return [3, 20];
                                case 19:
                                    e.listValue &&
                                        t.styles.listStyleType !== Rt.NONE &&
                                        ((this.ctx.font = this.createFontStyle(n)[0]),
                                        (this.ctx.fillStyle = ie(n.color)),
                                        (this.ctx.textBaseline = "middle"),
                                        (this.ctx.textAlign = "right"),
                                        (C = new c(
                                            t.bounds.left,
                                            t.bounds.top + $A(t.styles.paddingTop, t.bounds.width),
                                            t.bounds.width,
                                            (function (A, e) {
                                                return _A(A) && "normal" === A.value ? 1.2 * e : A.type === U.NUMBER_TOKEN ? e * A.number : kA(A) ? $A(A, e) : e;
                                            })(n.lineHeight, n.fontSize.number) /
                                                2 +
                                                1
                                        )),
                                        this.renderTextWithLetterSpacing(new Or(e.listValue, C), n.letterSpacing),
                                        (this.ctx.textBaseline = "bottom"),
                                        (this.ctx.textAlign = "left")),
                                        (a.label = 20);
                                case 20:
                                    return [2];
                            }
                        });
                    });
                }),
                (A.prototype.renderStackContent = function (A) {
                    return B(this, void 0, void 0, function () {
                        var e, t, r, n, s, i, o, B, c, l, Q, u, w, h, g;
                        return a(this, function (a) {
                            switch (a.label) {
                                case 0:
                                    return [4, this.renderNodeBackgroundAndBorders(A.element)];
                                case 1:
                                    a.sent(), (e = 0), (t = A.negativeZIndex), (a.label = 2);
                                case 2:
                                    return e < t.length ? ((g = t[e]), [4, this.renderStack(g)]) : [3, 5];
                                case 3:
                                    a.sent(), (a.label = 4);
                                case 4:
                                    return e++, [3, 2];
                                case 5:
                                    return [4, this.renderNodeContent(A.element)];
                                case 6:
                                    a.sent(), (r = 0), (n = A.nonInlineLevel), (a.label = 7);
                                case 7:
                                    return r < n.length ? ((g = n[r]), [4, this.renderNode(g)]) : [3, 10];
                                case 8:
                                    a.sent(), (a.label = 9);
                                case 9:
                                    return r++, [3, 7];
                                case 10:
                                    (s = 0), (i = A.nonPositionedFloats), (a.label = 11);
                                case 11:
                                    return s < i.length ? ((g = i[s]), [4, this.renderStack(g)]) : [3, 14];
                                case 12:
                                    a.sent(), (a.label = 13);
                                case 13:
                                    return s++, [3, 11];
                                case 14:
                                    (o = 0), (B = A.nonPositionedInlineLevel), (a.label = 15);
                                case 15:
                                    return o < B.length ? ((g = B[o]), [4, this.renderStack(g)]) : [3, 18];
                                case 16:
                                    a.sent(), (a.label = 17);
                                case 17:
                                    return o++, [3, 15];
                                case 18:
                                    (c = 0), (l = A.inlineLevel), (a.label = 19);
                                case 19:
                                    return c < l.length ? ((g = l[c]), [4, this.renderNode(g)]) : [3, 22];
                                case 20:
                                    a.sent(), (a.label = 21);
                                case 21:
                                    return c++, [3, 19];
                                case 22:
                                    (Q = 0), (u = A.zeroOrAutoZIndexOrTransformedOrOpacity), (a.label = 23);
                                case 23:
                                    return Q < u.length ? ((g = u[Q]), [4, this.renderStack(g)]) : [3, 26];
                                case 24:
                                    a.sent(), (a.label = 25);
                                case 25:
                                    return Q++, [3, 23];
                                case 26:
                                    (w = 0), (h = A.positiveZIndex), (a.label = 27);
                                case 27:
                                    return w < h.length ? ((g = h[w]), [4, this.renderStack(g)]) : [3, 30];
                                case 28:
                                    a.sent(), (a.label = 29);
                                case 29:
                                    return w++, [3, 27];
                                case 30:
                                    return [2];
                            }
                        });
                    });
                }),
                (A.prototype.mask = function (A) {
                    this.ctx.beginPath(),
                        this.ctx.moveTo(0, 0),
                        this.ctx.lineTo(this.canvas.width, 0),
                        this.ctx.lineTo(this.canvas.width, this.canvas.height),
                        this.ctx.lineTo(0, this.canvas.height),
                        this.ctx.lineTo(0, 0),
                        this.formatPath(A.slice(0).reverse()),
                        this.ctx.closePath();
                }),
                (A.prototype.path = function (A) {
                    this.ctx.beginPath(), this.formatPath(A), this.ctx.closePath();
                }),
                (A.prototype.formatPath = function (A) {
                    var e = this;
                    A.forEach(function (A, t) {
                        var r = ns(A) ? A.start : A;
                        0 === t ? e.ctx.moveTo(r.x, r.y) : e.ctx.lineTo(r.x, r.y), ns(A) && e.ctx.bezierCurveTo(A.startControl.x, A.startControl.y, A.endControl.x, A.endControl.y, A.end.x, A.end.y);
                    });
                }),
                (A.prototype.renderRepeat = function (A, e, t, r) {
                    this.path(A), (this.ctx.fillStyle = e), this.ctx.translate(t, r), this.ctx.fill(), this.ctx.translate(-t, -r);
                }),
                (A.prototype.resizeImage = function (A, e, t) {
                    if (A.width === e && A.height === t) return A;
                    var r = this.canvas.ownerDocument.createElement("canvas");
                    return (r.width = e), (r.height = t), r.getContext("2d").drawImage(A, 0, 0, A.width, A.height, 0, 0, e, t), r;
                }),
                (A.prototype.renderBackgroundImage = function (A) {
                    return B(this, void 0, void 0, function () {
                        var e, t, r, n, s, i;
                        return a(this, function (o) {
                            switch (o.label) {
                                case 0:
                                    (e = A.styles.backgroundImage.length - 1),
                                        (t = function (t) {
                                            var n, s, i, o, B, c, l, Q, u, w, h, g, U, C, d, E, F, f, H, p, N, K, m, I, T, v, L, R, y, b, O;
                                            return a(this, function (a) {
                                                switch (a.label) {
                                                    case 0:
                                                        if (t.type !== ge.URL) return [3, 5];
                                                        (n = void 0), (s = t.url), (a.label = 1);
                                                    case 1:
                                                        return a.trys.push([1, 3, , 4]), [4, r.options.cache.match(s)];
                                                    case 2:
                                                        return (n = a.sent()), [3, 4];
                                                    case 3:
                                                        return a.sent(), Te.getInstance(r.options.id).error("Error loading background-image " + s), [3, 4];
                                                    case 4:
                                                        return (
                                                            n &&
                                                                ((i = Cs(A, e, [n.width, n.height, n.width / n.height])),
                                                                (E = i[0]),
                                                                (K = i[1]),
                                                                (m = i[2]),
                                                                (H = i[3]),
                                                                (p = i[4]),
                                                                (C = r.ctx.createPattern(r.resizeImage(n, H, p), "repeat")),
                                                                r.renderRepeat(E, C, K, m)),
                                                            [3, 6]
                                                        );
                                                    case 5:
                                                        t.type === ge.LINEAR_GRADIENT
                                                            ? ((o = Cs(A, e, [null, null, null])),
                                                              (E = o[0]),
                                                              (K = o[1]),
                                                              (m = o[2]),
                                                              (H = o[3]),
                                                              (p = o[4]),
                                                              (B = Fe(t.angle, H, p)),
                                                              (c = B[0]),
                                                              (l = B[1]),
                                                              (Q = B[2]),
                                                              (u = B[3]),
                                                              (w = B[4]),
                                                              ((h = document.createElement("canvas")).width = H),
                                                              (h.height = p),
                                                              (g = h.getContext("2d")),
                                                              (U = g.createLinearGradient(l, u, Q, w)),
                                                              Ee(t.stops, c).forEach(function (A) {
                                                                  return U.addColorStop(A.stop, ie(A.color));
                                                              }),
                                                              (g.fillStyle = U),
                                                              g.fillRect(0, 0, H, p),
                                                              (C = r.ctx.createPattern(h, "repeat")),
                                                              r.renderRepeat(E, C, K, m))
                                                            : (function (A) {
                                                                  return A.type === ge.RADIAL_GRADIENT;
                                                              })(t) &&
                                                              ((d = Cs(A, e, [null, null, null])),
                                                              (E = d[0]),
                                                              (F = d[1]),
                                                              (f = d[2]),
                                                              (H = d[3]),
                                                              (p = d[4]),
                                                              (N = 0 === t.position.length ? [qA] : t.position),
                                                              (K = $A(N[0], H)),
                                                              (m = $A(N[N.length - 1], p)),
                                                              (I = (function (A, e, t, r, n) {
                                                                  var s = 0,
                                                                      i = 0;
                                                                  switch (A.size) {
                                                                      case Ve.CLOSEST_SIDE:
                                                                          A.shape === Pe.CIRCLE
                                                                              ? (s = i = Math.min(Math.abs(e), Math.abs(e - r), Math.abs(t), Math.abs(t - n)))
                                                                              : A.shape === Pe.ELLIPSE && ((s = Math.min(Math.abs(e), Math.abs(e - r))), (i = Math.min(Math.abs(t), Math.abs(t - n))));
                                                                          break;
                                                                      case Ve.CLOSEST_CORNER:
                                                                          if (A.shape === Pe.CIRCLE) s = i = Math.min(fe(e, t), fe(e, t - n), fe(e - r, t), fe(e - r, t - n));
                                                                          else if (A.shape === Pe.ELLIPSE) {
                                                                              var o = Math.min(Math.abs(t), Math.abs(t - n)) / Math.min(Math.abs(e), Math.abs(e - r)),
                                                                                  B = He(r, n, e, t, !0),
                                                                                  a = B[0],
                                                                                  c = B[1];
                                                                              i = o * (s = fe(a - e, (c - t) / o));
                                                                          }
                                                                          break;
                                                                      case Ve.FARTHEST_SIDE:
                                                                          A.shape === Pe.CIRCLE
                                                                              ? (s = i = Math.max(Math.abs(e), Math.abs(e - r), Math.abs(t), Math.abs(t - n)))
                                                                              : A.shape === Pe.ELLIPSE && ((s = Math.max(Math.abs(e), Math.abs(e - r))), (i = Math.max(Math.abs(t), Math.abs(t - n))));
                                                                          break;
                                                                      case Ve.FARTHEST_CORNER:
                                                                          if (A.shape === Pe.CIRCLE) s = i = Math.max(fe(e, t), fe(e, t - n), fe(e - r, t), fe(e - r, t - n));
                                                                          else if (A.shape === Pe.ELLIPSE) {
                                                                              o = Math.max(Math.abs(t), Math.abs(t - n)) / Math.max(Math.abs(e), Math.abs(e - r));
                                                                              var l = He(r, n, e, t, !1);
                                                                              (a = l[0]), (c = l[1]), (i = o * (s = fe(a - e, (c - t) / o)));
                                                                          }
                                                                  }
                                                                  return Array.isArray(A.size) && ((s = $A(A.size[0], r)), (i = 2 === A.size.length ? $A(A.size[1], n) : s)), [s, i];
                                                              })(t, K, m, H, p)),
                                                              (T = I[0]),
                                                              (v = I[1]),
                                                              T > 0 &&
                                                                  T > 0 &&
                                                                  ((L = r.ctx.createRadialGradient(F + K, f + m, 0, F + K, f + m, T)),
                                                                  Ee(t.stops, 2 * T).forEach(function (A) {
                                                                      return L.addColorStop(A.stop, ie(A.color));
                                                                  }),
                                                                  r.path(E),
                                                                  (r.ctx.fillStyle = L),
                                                                  T !== v
                                                                      ? ((R = A.bounds.left + 0.5 * A.bounds.width),
                                                                        (y = A.bounds.top + 0.5 * A.bounds.height),
                                                                        (O = 1 / (b = v / T)),
                                                                        r.ctx.save(),
                                                                        r.ctx.translate(R, y),
                                                                        r.ctx.transform(1, 0, 0, b, 0, 0),
                                                                        r.ctx.translate(-R, -y),
                                                                        r.ctx.fillRect(F, O * (f - y) + y, H, p * O),
                                                                        r.ctx.restore())
                                                                      : r.ctx.fill())),
                                                            (a.label = 6);
                                                    case 6:
                                                        return e--, [2];
                                                }
                                            });
                                        }),
                                        (r = this),
                                        (n = 0),
                                        (s = A.styles.backgroundImage.slice(0).reverse()),
                                        (o.label = 1);
                                case 1:
                                    return n < s.length ? ((i = s[n]), [5, t(i)]) : [3, 4];
                                case 2:
                                    o.sent(), (o.label = 3);
                                case 3:
                                    return n++, [3, 1];
                                case 4:
                                    return [2];
                            }
                        });
                    });
                }),
                (A.prototype.renderBorder = function (A, e, t) {
                    return B(this, void 0, void 0, function () {
                        return a(this, function (r) {
                            return (
                                this.path(
                                    (function (A, e) {
                                        switch (e) {
                                            case 0:
                                                return hs(A.topLeftBorderBox, A.topLeftPaddingBox, A.topRightBorderBox, A.topRightPaddingBox);
                                            case 1:
                                                return hs(A.topRightBorderBox, A.topRightPaddingBox, A.bottomRightBorderBox, A.bottomRightPaddingBox);
                                            case 2:
                                                return hs(A.bottomRightBorderBox, A.bottomRightPaddingBox, A.bottomLeftBorderBox, A.bottomLeftPaddingBox);
                                            case 3:
                                            default:
                                                return hs(A.bottomLeftBorderBox, A.bottomLeftPaddingBox, A.topLeftBorderBox, A.topLeftPaddingBox);
                                        }
                                    })(t, e)
                                ),
                                (this.ctx.fillStyle = ie(A)),
                                this.ctx.fill(),
                                [2]
                            );
                        });
                    });
                }),
                (A.prototype.renderNodeBackgroundAndBorders = function (A) {
                    return B(this, void 0, void 0, function () {
                        var e,
                            t,
                            r,
                            n,
                            s,
                            i,
                            o,
                            B,
                            c = this;
                        return a(this, function (a) {
                            switch (a.label) {
                                case 0:
                                    return (
                                        this.applyEffects(A.effects, 2),
                                        (e = A.container.styles),
                                        (t = !se(e.backgroundColor) || e.backgroundImage.length),
                                        (r = [
                                            { style: e.borderTopStyle, color: e.borderTopColor },
                                            { style: e.borderRightStyle, color: e.borderRightColor },
                                            { style: e.borderBottomStyle, color: e.borderBottomColor },
                                            { style: e.borderLeftStyle, color: e.borderLeftColor },
                                        ]),
                                        (n = ms(fs(e.backgroundClip, 0), A.curves)),
                                        t || e.boxShadow.length
                                            ? (this.ctx.save(), this.path(n), this.ctx.clip(), se(e.backgroundColor) || ((this.ctx.fillStyle = ie(e.backgroundColor)), this.ctx.fill()), [4, this.renderBackgroundImage(A.container)])
                                            : [3, 2]
                                    );
                                case 1:
                                    a.sent(),
                                        this.ctx.restore(),
                                        e.boxShadow
                                            .slice(0)
                                            .reverse()
                                            .forEach(function (e) {
                                                c.ctx.save();
                                                var t,
                                                    r,
                                                    n,
                                                    s,
                                                    i,
                                                    o = os(A.curves),
                                                    B = e.inset ? 0 : 1e4,
                                                    a =
                                                        ((t = o),
                                                        (r = -B + (e.inset ? 1 : -1) * e.spread.number),
                                                        (n = (e.inset ? 1 : -1) * e.spread.number),
                                                        (s = e.spread.number * (e.inset ? -2 : 2)),
                                                        (i = e.spread.number * (e.inset ? -2 : 2)),
                                                        t.map(function (A, e) {
                                                            switch (e) {
                                                                case 0:
                                                                    return A.add(r, n);
                                                                case 1:
                                                                    return A.add(r + s, n);
                                                                case 2:
                                                                    return A.add(r + s, n + i);
                                                                case 3:
                                                                    return A.add(r, n + i);
                                                            }
                                                            return A;
                                                        }));
                                                e.inset ? (c.path(o), c.ctx.clip(), c.mask(a)) : (c.mask(o), c.ctx.clip(), c.path(a)),
                                                    (c.ctx.shadowOffsetX = e.offsetX.number + B),
                                                    (c.ctx.shadowOffsetY = e.offsetY.number),
                                                    (c.ctx.shadowColor = ie(e.color)),
                                                    (c.ctx.shadowBlur = e.blur.number),
                                                    (c.ctx.fillStyle = e.inset ? ie(e.color) : "rgba(0,0,0,1)"),
                                                    c.ctx.fill(),
                                                    c.ctx.restore();
                                            }),
                                        (a.label = 2);
                                case 2:
                                    (s = 0), (i = 0), (o = r), (a.label = 3);
                                case 3:
                                    return i < o.length ? ((B = o[i]).style === Ze.NONE || se(B.color) ? [3, 5] : [4, this.renderBorder(B.color, s++, A.curves)]) : [3, 6];
                                case 4:
                                    a.sent(), (a.label = 5);
                                case 5:
                                    return i++, [3, 3];
                                case 6:
                                    return [2];
                            }
                        });
                    });
                }),
                (A.prototype.render = function (A) {
                    return B(this, void 0, void 0, function () {
                        var e;
                        return a(this, function (t) {
                            switch (t.label) {
                                case 0:
                                    return (
                                        this.options.backgroundColor &&
                                            ((this.ctx.fillStyle = ie(this.options.backgroundColor)), this.ctx.fillRect(this.options.x - this.options.scrollX, this.options.y - this.options.scrollY, this.options.width, this.options.height)),
                                        (r = new Qs(A, [])),
                                        (n = new ls(r)),
                                        us(r, n, n, (s = [])),
                                        ws(r.container, s),
                                        (e = n),
                                        [4, this.renderStack(e)]
                                    );
                                case 1:
                                    return t.sent(), this.applyEffects([], 2), [2, this.canvas];
                            }
                            var r, n, s;
                        });
                    });
                }),
                A
            );
        })(),
        Ks = function (A) {
            return A instanceof sn || A instanceof nn || (A instanceof rn && A.type !== en && A.type !== An);
        },
        ms = function (A, e) {
            switch (A) {
                case Qe.BORDER_BOX:
                    return os(e);
                case Qe.CONTENT_BOX:
                    return (function (A) {
                        return [A.topLeftContentBox, A.topRightContentBox, A.bottomRightContentBox, A.bottomLeftContentBox];
                    })(e);
                case Qe.PADDING_BOX:
                default:
                    return Bs(e);
            }
        },
        Is = function (A) {
            switch (A) {
                case zt.CENTER:
                    return "center";
                case zt.RIGHT:
                    return "right";
                case zt.LEFT:
                default:
                    return "left";
            }
        },
        Ts = (function () {
            function A(A) {
                (this.canvas = A.canvas ? A.canvas : document.createElement("canvas")),
                    (this.ctx = this.canvas.getContext("2d")),
                    (this.options = A),
                    (this.canvas.width = Math.floor(A.width * A.scale)),
                    (this.canvas.height = Math.floor(A.height * A.scale)),
                    (this.canvas.style.width = A.width + "px"),
                    (this.canvas.style.height = A.height + "px"),
                    this.ctx.scale(this.options.scale, this.options.scale),
                    this.ctx.translate(-A.x + A.scrollX, -A.y + A.scrollY),
                    Te.getInstance(A.id).debug("EXPERIMENTAL ForeignObject renderer initialized (" + A.width + "x" + A.height + " at " + A.x + "," + A.y + ") with scale " + A.scale);
            }
            return (
                (A.prototype.render = function (A) {
                    return B(this, void 0, void 0, function () {
                        var e, t;
                        return a(this, function (r) {
                            switch (r.label) {
                                case 0:
                                    return (
                                        (e = Ke(
                                            Math.max(this.options.windowWidth, this.options.width) * this.options.scale,
                                            Math.max(this.options.windowHeight, this.options.height) * this.options.scale,
                                            this.options.scrollX * this.options.scale,
                                            this.options.scrollY * this.options.scale,
                                            A
                                        )),
                                        [4, vs(e)]
                                    );
                                case 1:
                                    return (
                                        (t = r.sent()),
                                        this.options.backgroundColor && ((this.ctx.fillStyle = ie(this.options.backgroundColor)), this.ctx.fillRect(0, 0, this.options.width * this.options.scale, this.options.height * this.options.scale)),
                                        this.ctx.drawImage(t, -this.options.x * this.options.scale, -this.options.y * this.options.scale),
                                        [2, this.canvas]
                                    );
                            }
                        });
                    });
                }),
                A
            );
        })(),
        vs = function (A) {
            return new Promise(function (e, t) {
                var r = new Image();
                (r.onload = function () {
                    e(r);
                }),
                    (r.onerror = t),
                    (r.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(A)));
            });
        },
        Ls = function (A) {
            return ne(SA.create(A).parseComponentValue());
        },
        Rs = function (A, e) {
            return void 0 === e && (e = {}), bs(A, e);
        };
    ve.setContext(window);
    var ys,
        bs = function (A, e) {
            return B(void 0, void 0, void 0, function () {
                var t, r, n, s, i, B, Q, u, w, h, g, U, C, d, E, F, f, H, p, N, K, m, I;
                return a(this, function (a) {
                    switch (a.label) {
                        case 0:
                            if (!(t = A.ownerDocument)) throw new Error("Element is not attached to a Document");
                            if (!(r = t.defaultView)) throw new Error("Document is not attached to a Window");
                            return (
                                (n = (Math.round(1e3 * Math.random()) + Date.now()).toString(16)),
                                (s =
                                    fn(A) || "HTML" === A.tagName
                                        ? (function (A) {
                                              var e = A.body,
                                                  t = A.documentElement;
                                              if (!e || !t) throw new Error("Unable to get document size");
                                              var r = Math.max(Math.max(e.scrollWidth, t.scrollWidth), Math.max(e.offsetWidth, t.offsetWidth), Math.max(e.clientWidth, t.clientWidth)),
                                                  n = Math.max(Math.max(e.scrollHeight, t.scrollHeight), Math.max(e.offsetHeight, t.offsetHeight), Math.max(e.clientHeight, t.clientHeight));
                                              return new c(0, 0, r, n);
                                          })(t)
                                        : l(A)),
                                (i = s.width),
                                (B = s.height),
                                (Q = s.left),
                                (u = s.top),
                                (w = o({}, { allowTaint: !1, imageTimeout: 15e3, proxy: void 0, useCORS: !1 }, e)),
                                (h = {
                                    backgroundColor: "#ffffff",
                                    cache: e.cache ? e.cache : ve.create(n, w),
                                    logging: !0,
                                    removeContainer: !0,
                                    foreignObjectRendering: !1,
                                    scale: r.devicePixelRatio || 1,
                                    windowWidth: r.innerWidth,
                                    windowHeight: r.innerHeight,
                                    scrollX: r.pageXOffset,
                                    scrollY: r.pageYOffset,
                                    x: Q,
                                    y: u,
                                    width: Math.ceil(i),
                                    height: Math.ceil(B),
                                    id: n,
                                }),
                                (g = o({}, h, w, e)),
                                (U = new c(g.scrollX, g.scrollY, g.windowWidth, g.windowHeight)),
                                Te.create(n),
                                Te.getInstance(n).debug("Starting document clone"),
                                (C = new Pn(A, { id: n, onclone: g.onclone, ignoreElements: g.ignoreElements, inlineImages: g.foreignObjectRendering, copyStyles: g.foreignObjectRendering })),
                                (d = C.clonedReferenceElement) ? [4, C.toIFrame(t, U)] : [2, Promise.reject("Unable to find element in cloned iframe")]
                            );
                        case 1:
                            return (
                                (E = a.sent()),
                                (F = t.documentElement ? Ls(getComputedStyle(t.documentElement).backgroundColor) : he.TRANSPARENT),
                                (f = t.body ? Ls(getComputedStyle(t.body).backgroundColor) : he.TRANSPARENT),
                                (H = e.backgroundColor),
                                (p = "string" == typeof H ? Ls(H) : 4294967295),
                                (N = A === t.documentElement ? (se(F) ? (se(f) ? p : f) : F) : p),
                                (K = {
                                    id: n,
                                    cache: g.cache,
                                    backgroundColor: N,
                                    scale: g.scale,
                                    x: g.x,
                                    y: g.y,
                                    scrollX: g.scrollX,
                                    scrollY: g.scrollY,
                                    width: g.width,
                                    height: g.height,
                                    windowWidth: g.windowWidth,
                                    windowHeight: g.windowHeight,
                                }),
                                g.foreignObjectRendering ? (Te.getInstance(n).debug("Document cloned, using foreign object rendering"), [4, new Ts(K).render(d)]) : [3, 3]
                            );
                        case 2:
                            return (m = a.sent()), [3, 5];
                        case 3:
                            return (
                                Te.getInstance(n).debug("Document cloned, using computed rendering"),
                                ve.attachInstance(g.cache),
                                Te.getInstance(n).debug("Starting DOM parsing"),
                                (I = Qn(d)),
                                ve.detachInstance(),
                                N === I.styles.backgroundColor && (I.styles.backgroundColor = he.TRANSPARENT),
                                Te.getInstance(n).debug("Starting renderer"),
                                [4, new Ns(K).render(I)]
                            );
                        case 4:
                            (m = a.sent()), (a.label = 5);
                        case 5:
                            return (
                                !0 === g.removeContainer && (Os(E) || Te.getInstance(n).error("Cannot detach cloned iframe as it is not in the DOM anymore")),
                                Te.getInstance(n).debug("Finished rendering"),
                                Te.destroy(n),
                                ve.destroy(n),
                                [2, m]
                            );
                    }
                });
            });
        },
        Os = function (A) {
            return !!A.parentNode && (A.parentNode.removeChild(A), !0);
        };
    class Ss {
        constructor(e) {
            var t;
            (this.div = document.querySelector(`div.modal.${e}`)),
                (this.doneBtn = this.div.querySelector(".js-close")),
                (this.loader = this.div.querySelector(".js-loader")),
                null === (t = this.doneBtn) ||
                    void 0 === t ||
                    t.addEventListener("click", (A) => {
                        A.preventDefault(), this.hide();
                    }),
                this.div.addEventListener("click", (e) => {
                    e.target === this.div && (A.debug("modal:background.click"), this.hide());
                });
        }
        async show() {
            await this.performBeforeShow(), this.div.classList.add("fadeout"), this.div.classList.remove("hide"), await r.delay(100), this.div.classList.remove("fadeout"), await this.performShow(), this.loader.classList.add("hide");
        }
        async hide() {
            this.div.classList.add("fadeout"), await r.delay(200), this.div.classList.add("hide"), this.div.classList.remove("fadeout"), await this.performHide();
        }
    }
    class Ms extends Ss {
        constructor() {
            super("share-screenshot"), (this.imResult = this.div.querySelector(".js-result"));
        }
        async performBeforeShow() {
            this.loader.classList.remove("hide");
        }
        async performShow() {
            await r.delay(800);
            const A = document.querySelector("div.page-inner"),
                e = await Rs(A, {
                    logging: !1,
                    onclone: (A) => {
                        A.querySelector("div.page-inner").classList.add("html2canvas"), (A.querySelector("div.watermark").style.visibility = "visible");
                    },
                });
            this.imResult.src = e.toDataURL();
        }
        async performHide() {
            this.imResult.src = "";
        }
    }
    class Ds extends Ss {
        constructor(A) {
            super("save-publish"), (this.quill = A), (this.saveBtn = this.div.querySelector(".js-save")), (this.form = this.div.querySelector("form"));
            const e = Array.from(this.div.querySelectorAll(".doc-url"));
            for (const A of e)
                A.addEventListener("focus", () => {
                    A.select();
                });
            this.saveBtn.addEventListener("click", (A) => {
                A.preventDefault(), this.reportValidity() && this.save();
            }),
                this.form.addEventListener("submit", (A) => {
                    this.hideMentionList();
                    A.preventDefault(), this.saveBtn.click();
                });
        }
        async performBeforeShow() {}
        async performShow() {}
        async performHide() {}
        async reportValidity() {
            alert("At least ? characters must be entered");
            return false;
        }
        async save() {
            this.loader.classList.remove("hide");
            const t = this.div.querySelector(".doc-title").value,
                n = this.quill.getContents();
            A.log(JSON.stringify({ title: t, contents: n }));
            const s = await e.shared.postEdit({ title: t, contents: n });
            if ((await r.delay(800), s)) {
                this.loader.classList.add("hide"), this.hide();
                const A = this.div.querySelector(".doc-edit-url");
                window.location.href = A.value;
            } else window.alert("did not manage to save");
        }
    }
    !(function (A) {
        (A[(A.app = 0)] = "app"), (A[(A.landing = 1)] = "landing"), (A[(A.model = 2)] = "model");
    })(ys || (ys = {}));
    const _s = {
            page: document.body.classList.contains("app") ? ys.app : document.body.classList.contains("landing") ? ys.landing : ys.model,
            editable: "true" === document.body.dataset.editable,
            header: {
                shuffleBtn: document.querySelector("header .js-shuffle"),
                triggerBtn: document.querySelector("header .js-trigger"),
                mainInfoBtn: document.querySelector("header .title .info"),
                shareBtn: document.querySelector("header .js-share"),
                saveBtn: document.querySelector("header .js-save"),
                submitBtn: document.querySelector("header .js-submit"),
                duplicateBtn: document.querySelector("header .js-duplicate"),
            },
            shareScreenBtn: document.querySelector(".page-container .js-share"),
            loaderEditor: document.querySelector(".page-container .js-loader"),
            sliders: Array.from(document.querySelectorAll(".decoder-settings input.slider")),
            INITIAL_CONTENT: {},
            onLoad: (A, e) => {
                A === _s.page &&
                    document.addEventListener("DOMContentLoaded", () => {
                        e();
                    });
            },
        },
        xs = [
            "Before boarding your rocket to Mars, remember to pack these items",
            "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
            "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.",
            "Today, scientists confirmed the worst possible outcome: the massive asteroid will collide with Earth",
            "\n\t\tThor: The Tesseract belongs on Asgard, no human is a match for it.\n\t\tTony turns to leave, but Steve stops him.\n\t\tSteve: You're not going alone!\n\t\tTony: You gonna stop me?\n\t"
                .replace(/\t/g, "")
                .trim()
                .concat("\n"),
        ];
    _s.onLoad(ys.app, () => {
        var n, s, i;
        const o = new Ms(),
            B = { theme: void 0, modules: { toolbar: [], mention: {} } };
        _s.editable || (B.readOnly = !0);
        const a = new Quill("div.editor", B),
            c = a.getModule("mention");
        window.quill = a;
        const l = window.QUILL_C;
        l && a.setContents(l),
            a.container.appendChild(_s.loaderEditor),
            a.container.appendChild(_s.shareScreenBtn),
            // esti
            a.on('text-change', function(delta, oldDelta, source) {
                if (source == 'user') {
                    // console.log("A user action triggered this change.");
                    canceled = false;
                    _s.loaderEditor.classList.add("hide");
                        triggerPredictions(() => {
                        Q();
                    });
                }
            });
            if (TAB_ENABLED) {
                a.keyboard.addBinding({ key: t.Keys.TAB }, () => {
                    canceled = false;
                    Q();
                }),
                a.keyboard.bindings[t.Keys.TAB].unshift(a.keyboard.bindings[t.Keys.TAB].pop());
            }
        const submit = async () => {
            canceled = true;
            console.log('DEBUG:',e.shared);
            const text = a.editor.getText(0, Number.MAX_VALUE);
            console.log('Submitting...', text);
            if (text.length < MIN_TEXT_LENGTH) {
                alert('Too short - at least ' + MIN_TEXT_LENGTH + ' characters required');
            } else {
                
                e.shared.postSubmission(a.editor.getText(0, Number.MAX_VALUE));
                selections = [];
                suggestions = [];
                a.setText('');
                start_time = new Date().getTime()
                if (window.confirm(COMPLETION_MSG))
                {
                    window.location.href=COMPLETION_LINK;
                };
//                alert(COMPLETION_MSG);
            }
        };
        const Q = async () => {
            if (canceled || !window.completionEnabled) {
                return;
            }
            c.setCursorPos();
            const t = a.getBounds(c.getCursorPos());
            (_s.loaderEditor.style.top = `${t.top - 4}px`), (_s.loaderEditor.style.left = `${t.left + 4}px`), _s.loaderEditor.classList.remove("hide");
            const r = a.getText(0, c.getCursorPos());
            // TODO[dvs]: suppressed logging - A.debug("%c[About to launch autocomplete for]", "color: green;", r);
            if (r != "") {
                const n = await e.shared.postWithSettings({ context: r });
                _s.loaderEditor.classList.add("hide");
                // TODO[dvs]: suppressed logging - for (const e of n.sentences) A.log(e.value);
                if (REPORT_SELECTIONS_ON_SUBMIT) {
                    suggestions.push({userid: window.appUserId, timestamp: new Date().getTime(), input: r, option1: n.sentences[0].value, option2: n.sentences[1].value, option3: n.sentences[2].value, accepted: -1 });
                    c.trigger(n.sentences.map((A) => A.value));
                }
            }
        };
        if (
            (null === (n = _s.header.duplicateBtn) ||
                void 0 === n ||
                n.addEventListener("click", async (A) => {
                    A.preventDefault();
                    const t = await e.shared.postDuplicate();
                    window.location.href = t;
                }),
            !_s.editable)
        )
            return;
        const u = new Ds(a);
        _s.header.shuffleBtn.addEventListener("click", (A) => {
            A.preventDefault(), a.setText(r.randomItem(xs)), a.setSelection(a.getLength(), 0), Q();
        }),
            _s.header.submitBtn.addEventListener("click", (A) => {
                A.preventDefault(), submit(); // TODO[dvs]: implement submit
            }),
            _s.header.triggerBtn.addEventListener("click", (A) => {
                A.preventDefault(), Q();
            }),
            null === (s = _s.header.shareBtn) ||
                void 0 === s ||
                s.addEventListener("click", async (A) => {
                    A.preventDefault();
                    window.open(`https://twitter.com/share?url=${encodeURIComponent(window.location.href)}&text=${encodeURIComponent("Write With Transformer")}`);
                }),
            null === (i = _s.header.saveBtn) ||
                void 0 === i ||
                i.addEventListener("click", (A) => {
                    A.preventDefault(), c.hideMentionList(), u.show();
                }),
            _s.shareScreenBtn.addEventListener("click", async (A) => {
                A.preventDefault(), c.hideMentionList(), o.show();
            }),
            a.on("text-change", () => {
                _s.shareScreenBtn.classList.remove("hide");
                const A = a.getContents().ops.some((A) => A.attributes && !0 === A.attributes.bold);
                _s.shareScreenBtn.classList.toggle("fadeout", !A);
            }),
            document.addEventListener("click", (e) => {
                if (!(e.target instanceof HTMLAnchorElement && null !== e.target.closest("div.ql-editor"))) return;
                e.preventDefault(), e.stopPropagation();
                const t = e.target.getAttribute("href");
                A.debug("[click]", t), "#js-shuffle" === t ? _s.header.shuffleBtn.click() : window.open(e.target.href);
            }),
            document.addEventListener("scroll", (A) => {
                const e = document.getElementsByClassName("js-trigger")[0];
                scrollY > 100
                    ? ((e.style.position = "fixed"),
                      (e.style.top = "10px"),
                      (e.style.border = "1px solid blue"),
                      (e.style.backgroundColor = "white"),
                      (e.style.borderRadius = "100px"),
                      (e.style.padding = "5px"),
                      (e.style.zIndex = "1"),
                      (e.style.left = "50%"),
                      (e.style.transform = "translateX(-50%)"))
                    : ((e.style.position = "relative"),
                      (e.style.top = "auto"),
                      (e.style.border = "none"),
                      (e.style.backgroundColor = "white"),
                      (e.style.borderRadius = "0"),
                      (e.style.padding = "0"),
                      (e.style.zIndex = "1"),
                      (e.style.left = "auto"));
            });
        const w = (A) => {
            const e = A.parentNode.querySelector(".js-val"),
                t = Number.isInteger(A.valueAsNumber) ? A.valueAsNumber : Number(A.valueAsNumber.toFixed(2)),
                r = `value-${t}`;
            A.dataset[r] ? (e.innerText = A.dataset[r]) : (e.innerText = t.toString());
            const n = Number(A.getAttribute("min")),
                s = Number(A.getAttribute("max"));
            (e.className = t < n + (s - n) / 3 ? "js-val green" : t < n + (2 * (s - n)) / 3 ? "js-val orange" : "js-val red"),
                A.classList.contains("js-inverted") && (e.classList.contains("green") ? (e.classList.remove("green"), e.classList.add("red")) : e.classList.contains("red") && (e.classList.remove("red"), e.classList.add("green")));
        };
        for (const A of _s.sliders)
            w(A),
                A.addEventListener("input", () => {
                    w(A);
                });
    }),
        _s.onLoad(ys.landing, () => {
            n.init(document.querySelectorAll("[data-tilt]"), { glare: !0, scale: 1.06, "max-glare": 0.3, speed: 400 });
        });
})();
