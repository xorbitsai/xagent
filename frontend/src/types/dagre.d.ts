declare module 'dagre' {
  export interface GraphLabel {
    rankdir?: 'TB' | 'BT' | 'LR' | 'RL';
    nodesep?: number;
    ranksep?: number;
    marginx?: number;
    marginy?: number;
  }

  export interface Node {
    label?: string;
    width?: number;
    height?: number;
    x?: number;
    y?: number;
  }

  export interface Edge {
    label?: string;
    width?: number;
    height?: number;
  }

  export interface Graph {
    setGraph(obj: GraphLabel): void;
    setDefaultEdgeLabel(fn: () => string): void;
    setNode(id: string, obj: Node): void;
    setEdge(source: string, target: string, obj?: Edge): void;
    layout(): void;
    node(id: string): Node;
    edge(source: string, target: string): Edge;
    nodes(): Node[];
    edges(): Edge[];
    predecessors(id: string): Node[];
    successors(id: string): Node[];
  }

  export class Graph {
    constructor(obj?: GraphLabel);
    setGraph(obj: GraphLabel): void;
    setDefaultEdgeLabel(fn: () => string): void;
    setNode(id: string, obj: Node): void;
    setEdge(source: string, target: string, obj?: Edge): void;
    layout(): void;
    node(id: string): Node;
    edge(source: string, target: string): Edge;
    nodes(): Node[];
    edges(): Edge[];
    predecessors(id: string): Node[];
    successors(id: string): Node[];
  }

  export namespace graphlib {
    export class Graph {
      constructor(obj?: GraphLabel);
      setGraph(obj: GraphLabel): void;
      setDefaultEdgeLabel(fn: () => string): void;
      setNode(id: string, obj: Node): void;
      setEdge(source: string, target: string, obj?: Edge): void;
      layout(): void;
      node(id: string): Node;
      edge(source: string, target: string): Edge;
      nodes(): Node[];
      edges(): Edge[];
      predecessors(id: string): Node[];
      successors(id: string): Node[];
    }
  }

  const dagre: {
    graph: (obj?: GraphLabel) => Graph;
    graphlib: {
      Graph: typeof Graph;
    };
    layout: (graph: Graph) => void;
  };

  export default dagre;
}
