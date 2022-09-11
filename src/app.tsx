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

const MODEL_NAMES = ["gpt-j-6B", "gpt2-xl", "gpt2"];

const SCORES_URLS = new Map(MODEL_NAMES.map(modelName =>
    [
        modelName,
        TOKENS.map(
            (_, i) => `data/${modelName}.en_lines-ud-dev` +
                      `.${("0000" + i).slice(0, 5)}.xentdiff.npy`
        )
    ]
));

const npyjs = new Npyjs();

export class App extends React.Component {
    state = {docIndex: 0, model: MODEL_NAMES[0]};

    render() {
        return (
            <Card>
                <Card.Header>
                    <Row className="g-2">
                        <Col md={3}>
                            <FloatingLabel controlId="modelSelect" label="Model">
                                <Form.Select key="model"
                                             onChange={e => this.setState({model: e.target.value})}>
                                    {
                                        MODEL_NAMES.map(modelName => 
                                            <option key={modelName} value={modelName}>
                                                {modelName}
                                            </option>
                                        )
                                    }
                                </Form.Select>
                            </FloatingLabel>
                        </Col>
                        <Col md={9}>
                            <FloatingLabel controlId="docSelect" label="Text">
                                <Form.Select key="doc"
                                             onChange={e => this.setState({docIndex: parseInt(e.target.value)})}>
                                    {
                                        TOKENS.map((tokens, idx) => 
                                            <option key={idx} value={idx}>
                                                {[...tokens.slice(0, 20), "…"].join("")}
                                            </option>
                                        )
                                    }
                                </Form.Select>
                            </FloatingLabel>
                        </Col>
                    </Row>
                </Card.Header>
                <Card.Body>
                    <HighlightedText tokens={TOKENS[this.state.docIndex]}
                                     scoresUrl={SCORES_URLS.get(this.state.model)[this.state.docIndex]}
                                     key={`${this.state.model}:${this.state.docIndex}`} />
                </Card.Body>
            </Card>
        );
    }
}

type HighlightedTextProps = {
    tokens: string[],
    scoresUrl: string
};
type HighlightedTextState = {
    scores?: ndarray.NdArray<number[]>,
    activeIndex: number,
    isFrozen: boolean
};

class HighlightedText extends React.Component<HighlightedTextProps, HighlightedTextState> {
    state = {scores: null, activeIndex: null, isFrozen: false};

    constructor(props) {
        super(props);
        (async () => {
            const rawArray = await npyjs.load(props.scoresUrl);
            const scores = ndarray(rawArray.data as number[], rawArray.shape);
            this.setState({scores});
        })();
    }

    render() {
        const scores = this.getScores();
        let className = "highlighted-text";
        if (this.state.scores == null) {
            className += " loading";
        }

        const onClick = () => {
            this.setState({isFrozen: false});
        };

        return (
            <div className={className} onClick={onClick}>
            {
                this.props.tokens.map((t, i) => {
                    let className = "token";
                    if (this.state && this.state.activeIndex == i) {
                        className += " active";
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
                    };
                    const onClick = (event: React.MouseEvent) => {
                        this.setState({isFrozen: !this.state.isFrozen});
                        event.stopPropagation();
                        if (this.state.isFrozen) {  // setState is not in effect yet
                            this.setState({activeIndex: i});
                        }
                    };
                    return <span key={i} className={className} style={style}
                                 onMouseOver={onMouseOver} onClick={onClick}>{t}</span>;
                })
            }
            </div>
        );
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
}