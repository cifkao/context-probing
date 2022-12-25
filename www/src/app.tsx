import ndarray from "ndarray";
import ndarrayUnpack from "ndarray-unpack";
import Npyjs from "npyjs";
import React from "react";
import Card from "react-bootstrap/Card";
import Col from "react-bootstrap/Col";
import FloatingLabel from "react-bootstrap/FloatingLabel";
import Form from "react-bootstrap/Form";
import Row from "react-bootstrap/Row";

import TOKENS from "../data/en_lines-ud-dev.tokens.json";

const TEXT_NAMES = [
    "Microsoft Access manual",
    "City of Glass (Paul Auster)",
    "To Jerusalem and Back (Saul Bellow)",
    "Heart of Darkness (Joseph Conrad)",
    "Europarl",
    "A Guest of Honour (Nadine Gordimer)",
    "Harry Potter and the Chamber of Secrets (J. K. Rowling)",
    "Gut Symmetries (Jeanette Winterson)"
]

const MODEL_NAMES = ["gpt-j-6B", "gpt2-xl", "gpt2"];
const SCORE_TYPES = [["kldiff", "KL divergence"], ["xentdiff", "NLL loss"]];

const DATA_URLS = new Map(MODEL_NAMES.map(modelName =>
    [
        modelName,
        TOKENS.map(
            (_, i) => `data/${modelName}.en_lines-ud-dev` +
                      `.${("0000" + i).slice(0, 5)}`
        )
    ]
));

const VOCAB_URL = "data/inv_vocab.json";
let vocab = {};

function getDataUrl(modelName: string, docIndex: number, key: string) {
    return DATA_URLS.get(modelName)[docIndex] + `.${key}.npy`;
}

const npyjs = new Npyjs();

async function loadNumpy(url: string) {
    const rawArray = await npyjs.load(url);
    return ndarray(rawArray.data as number[], rawArray.shape);
}

export class App extends React.Component {
    state = {docIndex: 0, model: MODEL_NAMES[0], scoreType: SCORE_TYPES[0][0], showTopk: false, isFrozen: false};

    render() {
        return (
            <Card>
                <Card.Header>
                    <Row className="g-2">
                        <Col md={2}>
                            <FloatingLabel controlId="modelSelect" label="Model">
                                <Form.Select key="model"
                                             onChange={e => this.setState({model: e.target.value})}>
                                    {
                                        MODEL_NAMES.map(key =>
                                            <option key={key} value={key}>{key}</option>
                                        )
                                    }
                                </Form.Select>
                            </FloatingLabel>
                        </Col>
                        <Col md={2}>
                            <FloatingLabel controlId="scoreSelect" label="Metric">
                                <Form.Select key="scoreType"
                                             onChange={e => this.setState({scoreType: e.target.value})}>
                                    {
                                        SCORE_TYPES.map(([key, text]) =>
                                            <option key={key} value={key}>{text}</option>
                                        )
                                    }
                                </Form.Select>
                            </FloatingLabel>
                        </Col>
                        <Col md={8}>
                            <FloatingLabel controlId="docSelect" label="Text">
                                <Form.Select key="doc"
                                             onChange={e => this.setState({docIndex: parseInt(e.target.value)})}>
                                    {
                                        TOKENS.map((tokens, idx) =>
                                            <option key={idx} value={idx}>{TEXT_NAMES[idx]}</option>
                                        )
                                    }
                                </Form.Select>
                            </FloatingLabel>
                        </Col>
                    </Row>
                    <Row className="g-2 mt-1">
                        <Col>
                            <div className={this.state.isFrozen && !this.state.showTopk ? "nudge" : ""}>
                                <Form.Check type="switch" id="topkSwitch"
                                    label="Show top predictions (loads 40MB+ of data)"
                                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => this.setState({showTopk: e.target.checked})} />
                            </div>
                        </Col>
                    </Row>
                </Card.Header>
                <Card.Body>
                    <HighlightedText tokens={TOKENS[this.state.docIndex]}
                                     scoresUrl={getDataUrl(this.state.model, this.state.docIndex, this.state.scoreType)}
                                     topkUrl={this.state.showTopk ? getDataUrl(this.state.model, this.state.docIndex, "topk") : null}
                                     key={`${this.state.model}:${this.state.docIndex}:${this.state.scoreType}:${this.state.showTopk}`}
                                     onFrozenChange={(isFrozen: boolean) => { this.setState({isFrozen: isFrozen}); }} />
                </Card.Body>
            </Card>
        );
    }
}

type HighlightedTextProps = {
    tokens: string[],
    scoresUrl: string,
    topkUrl: string,
    onFrozenChange: (isFrozen: boolean) => void
};
type HighlightedTextState = {
    scores: ndarray.NdArray<number[]>,
    topk: ndarray.NdArray<number[]>,
    activeIndex: number,
    hoverIndex: number,
    isFrozen: boolean
};

class HighlightedText extends React.Component<HighlightedTextProps, HighlightedTextState> {
    state = {scores: null, topk: null, activeIndex: null, hoverIndex: null, isFrozen: false};

    constructor(props) {
        super(props);
        (async () => {
            const scores = loadNumpy(props.scoresUrl);
            const topk = props.topkUrl ? loadNumpy(props.topkUrl) : null;
            if (topk && Object.keys(vocab).length == 0) {
                Object.assign(vocab, await (await fetch(VOCAB_URL)).json());
            }
            this.setState({scores: await scores, topk: await topk});
        })();
    }

    render() {
        const scores = this.getScores();
        const topk = this.getTopk();

        let className = "highlighted-text";
        if (this.state.scores == null) {
            className += " loading";
        }

        const onClick = () => {
            this.setState({isFrozen: false});
        };

        return <>
            <div className="status-bar" key="status-bar">
                <span className={this.state.isFrozen ? "" : " d-none"}><i className="fa fa-lock"></i> </span>
                <strong>target:</strong>
                {
                    this.state.activeIndex != null
                    ? <span className="token">{this.props.tokens[this.state.activeIndex]}</span>
                    : <></>
                }
                {
                    this.state.hoverIndex != null && topk[this.state.hoverIndex] != null
                    ? <>
                        <strong> top:</strong>
                        {topk[this.state.hoverIndex].map(token => <span className="token">{token}</span>)}
                    </>
                    : <></>
                }
            </div>
            <div className={className} onClick={onClick} key="text">
            {
                this.props.tokens.map((t, i) => {
                    let className = "token";
                    if (this.state) {
                        if (this.state.activeIndex == i) {
                            className += " active";
                        }
                        if (i >= this.state.hoverIndex && i < this.state.activeIndex && this.props.topkUrl != null) {
                            className += " context";
                        }
                    }
                    const style = {
                        backgroundColor:
                            scores[i] > 0
                            ? `rgba(255, 32, 32, ${scores[i]})`
                            : `rgba(32, 255, 32, ${-scores[i]})`
                    };

                    const onMouseOver = () => {
                        if (!this.state.isFrozen) {
                            this.setState({activeIndex: i});
                        }
                        this.setState({hoverIndex: i});
                    };
                    const onClick = (event: React.MouseEvent) => {
                        this.setState({isFrozen: !this.state.isFrozen});
                        // setState is not yet in effect below
                        this.props.onFrozenChange(!this.state.isFrozen);
                        if (this.state.isFrozen) {
                            this.setState({activeIndex: i});
                        }
                        event.stopPropagation();
                    };
                    return <span key={i} className={className} style={style}
                                 onMouseOver={onMouseOver} onClick={onClick}>{t}</span>;
                })
            }
            </div>
        </>;
    }

    private getScores() {
        if (!this.state || !this.state.scores || this.state.activeIndex == null) {
            return this.props.tokens.map(() => 0);
        }

        const i = this.state.activeIndex;
        const hi = Math.min(Math.max(0, i - 1), this.state.scores.shape[1]);
        const row = ndarrayUnpack(
            this.state.scores.pick(i).hi(hi).step(-1)
        ).map(x => x / 127) as number[];
        let result = [
            ...Array(Math.max(0, i - 1 - row.length)).fill(0), 
            ...row.map((x) => x == undefined || isNaN(x) ? 0 : x)
        ];
        result = [...result, ...Array(this.props.tokens.length - result.length).fill(0)];
        return result;
    }

    private getTopk() {
        if (!this.state || !this.state.topk || this.state.activeIndex == null) {
            return this.props.tokens.map(() => null);
        }

        const i = this.state.activeIndex;
        const hi = Math.min(Math.max(0, i), this.state.topk.shape[1]);
        const row = ndarrayUnpack(
            this.state.topk.pick(i).hi(hi).step(-1)
        ).map(l => l.map(i => vocab[i])) as string[][];
        let result = [
            ...Array(Math.max(0, i - row.length)).fill(null),
            ...row
        ];
        result = [...result, ...Array(this.props.tokens.length - result.length).fill(null)];
        return result;
    }
}
