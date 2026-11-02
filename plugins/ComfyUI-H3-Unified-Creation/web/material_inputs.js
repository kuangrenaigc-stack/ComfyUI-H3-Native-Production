import { app } from '../../scripts/app.js';
import { configureReferenceInputs } from './reference_inputs.js';

app.registerExtension({
  name:'H3.UnifiedCreationV2.MaterialSliders',
  async beforeRegisterNodeDef(nodeType,nodeData) {
    if (nodeData.name!=='H3UnifiedMaterialsV2') return;
    for (const hook of ['onNodeCreated','onConfigure']) {
      const previous=nodeType.prototype[hook];
      nodeType.prototype[hook]=function(...args) {
        const result=previous?.apply(this,args);
        configureReferenceInputs(this);
        return result;
      };
    }
  }
});
