use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

/**
 * A tree structure
 *
 * instead of linking nodes directly,
 * data is contained in a single arena or vector
 * this way there is a single reference
 * nodes instead contain the index of their parent and children in the arena
 */
pub type NodeId = usize;

#[derive(Debug)]
pub struct Tree<T> {
    nodes: Vec<Node<T>>
}

#[derive(Debug)]
pub struct Node<T> {
    pub children: Vec<NodeId>,
    parent: Option<NodeId>,
    pub data: T
}

impl<T> Node<T> {
    pub fn new(parent: Option<NodeId>, data: T) -> Self {
        Node {
            data: data,
            parent: parent,
            children: Vec::new()
        }
    }
    pub fn set_parent(&mut self, parent: NodeId) {
        self.parent = Some(parent);
    }
    pub fn add_child(&mut self, child: NodeId) {
        self.children.push(child);
    }
}

impl<T> Tree<T> {
    pub fn new() -> Tree<T> {
        Tree {
            nodes: Vec::new()
        }
    }
    pub fn create_node(&mut self, parent: Option<NodeId>, data: T) -> NodeId {
        let index: NodeId = self.nodes.len();
        let node = Node::new(parent, data);
        self.nodes.push(node);
        return index;
    }
    pub fn get_node_mut(&mut self, idx: NodeId) -> &mut Node<T> {
        return &mut self.nodes[idx];
    }
    pub fn get_node(&self, idx: NodeId) -> &Node<T> {
        return &self.nodes[idx];
    }
    // returns a recursive generator for node a specified node
    pub fn generator(&self, node: NodeId) -> Box<dyn Generator<Yield = &T, Return = ()> + '_> {
        Box::new(move || {
            let n = self.get_node(node);
            yield &n.data;
            for i in &n.children {
                let mut subgen = Box::into_pin(self.generator(*i));
                loop {
                    match subgen.as_mut().resume(()) {
                        GeneratorState::Yielded(data) => { yield data; },
                        GeneratorState::Complete(_) => { break; }
                    }
                }
            }
        })
    }
}
