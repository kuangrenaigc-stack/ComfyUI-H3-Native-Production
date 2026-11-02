import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "H3.UnifiedCreationV2.Preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3UnifiedPreviewV2") return;
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = previous?.apply(this, arguments);
            if (!this._h3V2Preview) {
                const widget = ComfyWidgets.STRING(this, "解析预览", ["STRING", { multiline: true }], app).widget;
                widget.inputEl.readOnly = true;
                widget.options = { ...widget.options, serialize: false };
                widget.serializeValue = async () => undefined;
                this._h3V2Preview = widget;
            }
            this._h3V2Preview.value = (message.text ?? []).join("\n\n");
            this.setSize([Math.max(420, this.size[0]), Math.max(300, this.size[1])]);
            this.setDirtyCanvas(true, true);
            return result;
        };
    },
});
