import ndarray from "ndarray";
import ndarrayUnpack from "ndarray-unpack";
import Npyjs from "npyjs";
import React from "react";

import TOKENS from "../data/en_lines-ud-dev.tokens.json";

const SCORES_URLS = TOKENS.map(
    (_, i) => `data/gpt-j-6B.en_lines-ud-dev.${("0000" + i).slice(0, 5)}.xentdiff.npy`
);

export class App extends React.Component {
    scoresPromises: Promise<ndarray.NdArray<number[]>>[];

    constructor(props: {}) {
        super(props);

        const npyjs = new Npyjs();
        this.scoresPromises = SCORES_URLS.map(async (url) => {
            const rawArray = await npyjs.load(url);
            return ndarray(rawArray.data as number[], rawArray.shape);
        });
    }

    render() {
        return (
            <div>
            {
                TOKENS.map((t, i) => (
                    <HighlightedText key={i} tokens={t}
                                     scoresPromise={this.scoresPromises[i]} />
                ))
            }
            </div>
        );
    }
}

type HighlightedTextProps = {
    tokens: string[],
    scoresPromise: Promise<ndarray.NdArray<number[]>>
};
type HighlightedTextState = {
    activeIndex: number
};

class HighlightedText extends React.Component<HighlightedTextProps, HighlightedTextState> {
    scores?: ndarray.NdArray<number[]>;

    constructor(props: HighlightedTextProps) {
        super(props);
        (async () => {
            this.scores = await props.scoresPromise;
        })();
    }

    render() {
        const scores = this.getScores();
        return (
            <div>
            {
                this.props.tokens.map((t, i) => {
                    const style = {
                        backgroundColor:
                            scores[i] > 0
                            ? `rgba(255, 32, 32, ${scores[i]})`
                            : `rgba(32, 255, 32, ${-scores[i]})`,
                        outline: this.state && this.state.activeIndex == i ? "1px solid black" : null
                    };
                    const onMouseOver = () => {
                        this.setState({activeIndex: i});
                    };
                    return <span key={i} style={style} onMouseOver={onMouseOver}>{t}</span>;
                })
            }
            <hr />
            </div>
        );
    }

    private getScores() {
        if (!this.scores || !this.state) {
            return this.props.tokens.map(() => 0);
        }

        const i = this.state.activeIndex;
        const hi = Math.min(Math.max(0, i - 1), this.scores.shape[1]);
        const row = ndarrayUnpack(
            this.scores.pick(i).hi(hi).step(-1)
        ).map(x => x / 127) as number[];
        let result = [
            ...Array(Math.max(0, i - 1 - row.length)).fill(0), 
            ...row.map((x) => x == undefined || isNaN(x) ? 0 : x)
        ];
        result = [...result, ...Array(this.props.tokens.length - result.length).fill(0)];
        return result;
    }
}