import type { NodePath, PluginObj, Visitor } from "@babel/core";
import type {
  AwaitExpression,
  CallExpression,
  Expression,
  FunctionDeclaration,
  Identifier,
  MemberExpression,
  Node,
  ObjectExpression,
  ObjectProperty,
  Program,
  StringLiteral,
} from "@babel/types";
import { RuleSyntaxError } from "./errors.js";
import { checkIdentifier, IdElement } from "./identifier.js";
import { Category } from "./types.js";

type BabelExport = typeof import("@babel/core");
type ThisEnv =
  | {
      category?: Category;
    }
  | undefined;

const enterIgnorance = Symbol("enterIgnorance");

export default function ruleTranspilerPlugin(this: ThisEnv, babel: BabelExport): PluginObj {
  const pluginThis = this;
  const { types: t } = babel;

  /**
   * Transform a JS object to Babel AST, in literal form.
   * @param literal
   * @param preserveAst Preserve AST Node in any component
   * @returns
   */
  function astify(literal: unknown, preserveAst = false): Expression {
    // Dirty hack here
    function isNode(l: any): l is Expression {
      if (l?.constructor?.name === "Node") return true;
      return (
        typeof l?.type === "string" &&
        (l.type.endsWith("Expression") || l.type.endsWith("Literal"))
      );
    }
    if (preserveAst && isNode(literal)) {
      return literal;
    }
    if (literal === null) {
      return t.nullLiteral();
    }
    switch (typeof literal) {
      case "number":
        return t.numericLiteral(literal);
      case "string":
        return t.stringLiteral(literal);
      case "boolean":
        return t.booleanLiteral(literal);
      case "undefined":
        return t.unaryExpression("void", t.numericLiteral(0), true);
      case "bigint":
        return t.bigIntLiteral(literal.toString());
      case "function":
      case "symbol":
        throw new Error(`${typeof literal} not supported`);
      default:
        if (Array.isArray(literal)) {
          return t.arrayExpression(literal.map((l) => astify(l, preserveAst)));
        }
        if (literal instanceof RegExp) {
          return t.regExpLiteral(literal.source, literal.flags);
        }

        const lit = literal as Record<string, unknown>;
        return t.objectExpression(
          Object.entries(lit)
            .filter(([k]) => typeof k === "string")
            .map(([k, v]): ObjectProperty => {
              return t.objectProperty(
                t.isValidIdentifier(k) ? t.identifier(k) : t.stringLiteral(k),
                astify(v, preserveAst)
              );
            })
        );
    }
  }

  /**
   * ??????????????????????????????????????? `this.target` ??????
   */
  interface IdVisitorState {
    readonly controller: RuleController;

    /**
     * ?????????????????????????????????
     * ??????????????????????????????????????????????????????
     */
    target: IdElement[] | null;

    /**
     * ?????????????????????????????????????????????????????????????????????????????????????????????
     * ??????????????????????????????/?????????????????????????????????????????????
     * ???????????????????????????  
     * ????????????????????????????????????
     */
    ignoreDepth: number;
  };

  type TopVisitorState = {
    readonly controller: RuleController;
  };
  class RuleController {
    #controllerId: Identifier;

    constructor(public readonly category: Category, path: NodePath<Program>) {
      const controllerClassId = path.scope.generateUidIdentifier("Controller");
      this.#controllerId = path.scope.generateUidIdentifier("controller");
      const importDecl = t.importDeclaration(
        [t.importSpecifier(controllerClassId, t.identifier("Controller"))],
        t.stringLiteral("graduate")
      );
      const initDecl = t.variableDeclaration("const", [
        t.variableDeclarator(
          this.#controllerId,
          t.newExpression(controllerClassId, [t.stringLiteral(category)])
        ),
      ]);
      const exportDecl = t.exportDefaultDeclaration(this.#controllerId);
      path.node.body.splice(0, 0, importDecl, initDecl);
      path.node.body.push(exportDecl);
    }
    private method(name: string) {
      return t.memberExpression(this.#controllerId, t.identifier(name));
    }
    readonly assert = () => this.method("assert");
    readonly send = () => this.method("send");
    readonly addCase = () => this.method("addCase");
  }

  /**
   * ??? AST ??????????????????????????? awaited-????????????
   * @param this 
   * @param path 
   */
  function transformId(this: IdVisitorState, path: NodePath<Node>) {
    // ?????????????????????????????????????????????????????????????????????????????????????????????????????????
    if (this.target && !this.ignoreDepth) {
      const node = path.node;
      if (node.type === "MemberExpression") {
        const prop = node.property;
        if (prop.type !== "Identifier") {
          throw new RuleSyntaxError(
            `Non-identifier member '${prop.type}' not supported.`,
            prop
          );
        }
        this.target.push({
          type: "identifier",
          name: prop.name,
          start: prop.start,
          end: prop.end,
        });
      } else if (node.type === "CallExpression") {
        this.target.push({
          type: "call",
          args: node.arguments.filter((n): n is Expression =>
            t.isExpression(n)
          ),
          start: node.start,
          end: node.end,
        });
      }
      // ???????????????????????????????????????????????????????????????????????????????????????????????????
      // ????????????????????? awaited-????????????
      if (
        path.parent.type !== "MemberExpression" &&
        !(path.parent.type === "CallExpression" && path.parent.callee === node)
      ) {
        const production = checkIdentifier(
          this.controller.category,
          this.target
        );
        this.target = null;

        if (path.parent.type !== "AwaitExpression") {
          path.replaceWith(
            t.awaitExpression(
              t.callExpression(this.controller.send(), [
                astify(production, true),
              ])
            )
          );
        }
      }
    }
  }
  const idVisitor: Visitor<IdVisitorState> = {
    enter(path) {
      // ???????????????????????????????????????
      if (
        this.target &&
        path.parent.type === "CallExpression" &&
        path.parent.callee !== path.node
      ) {
        // ???????????????????????????????????????????????????????????????????????????
        // ??????????????????????????????
        // ????????????????????????????????????
        if (!Object.hasOwn(path.parent, enterIgnorance)) {
          Object.defineProperty(path.parent, enterIgnorance, { value: true, enumerable: true, configurable: true });
          this.ignoreDepth++;
        }
        const subThis = {
          controller: this.controller,
          target: null,
          ignoreDepth: 0,
        };
        // ?????????????????????????????????????????????????????????
        path.traverse(idVisitor, subThis);
        // .traverse ????????????????????????????????????????????????
        transformId.call(subThis, path);
      }
    },
    exit(path) {
      // ??????????????????????????????????????????
      if (path.node.type === "CallExpression" || path.node.type === "MemberExpression" || path.node.type === "Identifier") {
        transformId.call(this, path);
      }
      // ????????????????????????????????????????????????
      if (Object.hasOwn(path.parent, enterIgnorance)) {
        this.ignoreDepth--;
        // @ts-expect-error No symbol typing
        delete path.parent[enterIgnorance];
      }
    },
    // ?????????????????????????????????????????????????????????????????????????????????
    Identifier: {
      enter(path) {
        const { start, end, name } = path.node;
        if (path.isReferencedIdentifier() && !path.scope.hasBinding(name)) {
          this.target = [
            {
              type: "identifier",
              name,
              start,
              end,
            },
          ];
        }
      },
    },
  };
  const topVisitor: Visitor<TopVisitorState> = {
    BlockStatement(path) {
      if (path.parent.type !== "Program") return;
      path.traverse(idVisitor, {
        target: null,
        controller: this.controller,
        ignoreDepth: 0,
      });
      const caseId = path.scope.generateUidIdentifier();
      path.replaceWith(
        t.functionDeclaration(caseId, [], path.node, false, true)
      );
      path.insertAfter(
        t.expressionStatement(
          t.callExpression(this.controller.addCase(), [caseId])
        )
      );
    },
    LabeledStatement: {
      exit(path) {
        const { node } = path;
        if (node.label.name !== "assert") return;
        const statement = node.body;
        if (statement.type !== "ExpressionStatement") {
          throw new RuleSyntaxError("Assertion not an expression", statement);
        }
        const { expression } = statement;
        const offset = (expression.start ?? 0) - (node.start ?? 0);
        const length = (expression.end ?? 1) - (expression.start ?? 0);
        const src = path.getSource().substring(offset, offset + length);

        let arg: Expression;
        if (expression.type === "BinaryExpression") {
          const { left, right } = expression;
          if (left.type === "PrivateName") {
            throw new RuleSyntaxError("Why private name here?", left);
          }
          arg = astify(
            {
              type: t.stringLiteral("binary"),
              src: t.stringLiteral(src),
              op: t.stringLiteral(expression.operator),
              lhs: left,
              rhs: right,
            },
            true
          );
        } else {
          arg = astify(
            {
              type: t.stringLiteral("other"),
              src: t.stringLiteral(src),
              expr: expression,
            },
            true
          );
        }
        path.replaceWith(t.callExpression(this.controller.assert(), [arg]));
      },
    },
  };
  return {
    visitor: {
      Program(path) {
        const { body, directives } = path.node;
        if (directives.length === 0) {
          throw new RuleSyntaxError(`No "use ..." directive found.`, path.node);
        }
        if (directives.length > 1) {
          throw new RuleSyntaxError(
            `Multiple directives. The first directive is "${directives[0].value.value}"`,
            directives[1]
          );
        }
        let category: Category;
        if (directives[0].value.value === "use web") {
          category = "web";
        } else if (directives[0].value.value === "use form") {
          category = "form";
        } else if (directives[0].value.value === "use graphics.turtle") {
          category = "graphics.turtle";
        } else {
          throw new RuleSyntaxError(
            `Unknown rule directive ${directives[0].value.value}`,
            directives[0]
          );
        }
        pluginThis && (pluginThis.category = category);
        if (body.length === 0) {
          throw new RuleSyntaxError("Empty rule", path.node);
        }
        for (const stmt of body) {
          if (stmt.type !== "BlockStatement" && !t.isDeclaration(stmt)) {
            throw new RuleSyntaxError(
              `A block or declaration expected at top-level, ${stmt.type} found`,
              stmt
            );
          }
        }
        path.node.directives = [];
        const controller = new RuleController(category, path);
        path.traverse(topVisitor, {
          controller,
        });
      },
    },
  };
}
