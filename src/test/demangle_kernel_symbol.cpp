#include <cxxabi.h>
#include <exception>
#include <iostream>
#include <vector>

#include "llvmdemangler/ItaniumDemangle.h"

constexpr const char *itanium_demangle::FloatData<float>::spec;
constexpr const char *itanium_demangle::FloatData<double>::spec;
constexpr const char *itanium_demangle::FloatData<long double>::spec;

class BumpPointerAllocator {
    struct BlockMeta {
        BlockMeta *Next;
        size_t Current;
    };

    static constexpr size_t AllocSize = 4096;
    static constexpr size_t UsableAllocSize = AllocSize - sizeof(BlockMeta);

    alignas(long double) char InitialBuffer[AllocSize];
    BlockMeta *BlockList = nullptr;

    void grow() {
        char *NewMeta = static_cast<char *>(std::malloc(AllocSize));
        if (NewMeta == nullptr)
            std::terminate();
        BlockList = new (NewMeta) BlockMeta{BlockList, 0};
    }

    void *allocateMassive(size_t NBytes) {
        NBytes += sizeof(BlockMeta);
        BlockMeta *NewMeta = reinterpret_cast<BlockMeta *>(std::malloc(NBytes));
        if (NewMeta == nullptr)
            std::terminate();
        BlockList->Next = new (NewMeta) BlockMeta{BlockList->Next, 0};
        return static_cast<void *>(NewMeta + 1);
    }

  public:
    BumpPointerAllocator()
        : BlockList(new (InitialBuffer) BlockMeta{nullptr, 0}) {}

    void *allocate(size_t N) {
        N = (N + 15u) & ~15u;
        if (N + BlockList->Current >= UsableAllocSize) {
            if (N > UsableAllocSize)
                return allocateMassive(N);
            grow();
        }
        BlockList->Current += N;
        return static_cast<void *>(reinterpret_cast<char *>(BlockList + 1) +
                                   BlockList->Current - N);
    }

    void reset() {
        while (BlockList) {
            BlockMeta *Tmp = BlockList;
            BlockList = BlockList->Next;
            if (reinterpret_cast<char *>(Tmp) != InitialBuffer)
                std::free(Tmp);
        }
        BlockList = new (InitialBuffer) BlockMeta{nullptr, 0};
    }

    ~BumpPointerAllocator() { reset(); }
};

class DefaultAllocator {
    BumpPointerAllocator Alloc;

  public:
    void reset() { Alloc.reset(); }

    template <typename T, typename... Args> T *makeNode(Args &&...args) {
        return new (Alloc.allocate(sizeof(T))) T(std::forward<Args>(args)...);
    }

    void *allocateNodeArray(size_t sz) {
        return Alloc.allocate(sizeof(itanium_demangle::Node *) * sz);
    }
};

const char *itanium_demangle::parse_discriminator(const char *first,
                                                  const char *last) {
    // parse but ignore discriminator
    if (first != last) {
        if (*first == '_') {
            const char *t1 = first + 1;
            if (t1 != last) {
                if (std::isdigit(*t1))
                    first = t1 + 1;
                else if (*t1 == '_') {
                    for (++t1; t1 != last && std::isdigit(*t1); ++t1)
                        ;
                    if (t1 != last && *t1 == '_')
                        first = t1 + 1;
                }
            }
        } else if (std::isdigit(*first)) {
            const char *t1 = first + 1;
            for (; t1 != last && std::isdigit(*t1); ++t1)
                ;
            if (t1 == last)
                first = last;
        }
    }
    return first;
}

template <class StringLikeA, class StringLikeB>
bool iequals(const StringLikeA &a, const StringLikeB &b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char a, char b) { return tolower(a) == tolower(b); });
}

void deduce_nested_ptr(itanium_demangle::NameWithTemplateArgs *tmp_node,
                       int &is_ptr) {
    assert(tmp_node != nullptr);

    // If tempplate is an array
    if (iequals(tmp_node->Name->getBaseName(), std::string("array"))) {
        auto template_args = dynamic_cast<itanium_demangle::TemplateArgs *>(
            tmp_node->TemplateArgs);
        if (!template_args)
            throw std::runtime_error("Array type not supported.");
        if (template_args->getParams().size() != 2)
            throw std::runtime_error("Array type not supported.");

        if (template_args->getParams()[0]->getKind() ==
            itanium_demangle::Node::KPointerType) {
            is_ptr = true;
        } else {
            is_ptr = false;
        }
    }
}

using Demangler = itanium_demangle::ManglingParser<DefaultAllocator>;

std::vector<itanium_demangle::Node *>
expand_mangled_par(itanium_demangle::FunctionEncoding* array) {
    size_t N_unexpanded_par = array->getParams().size();
    std::vector<itanium_demangle::Node *> expanded(N_unexpanded_par);
    unsigned cur_idx = 0;

    for(unsigned i = 0; i < N_unexpanded_par; ++i) {
        if(array->getParams()[i]->getKind() == itanium_demangle::Node::KParameterPackExpansion) {
            auto pack = dynamic_cast<const itanium_demangle::ParameterPack *>(
                            dynamic_cast<itanium_demangle::ParameterPackExpansion *>(array->getParams()[i])
                                ->getChild())->getPack();

            expanded.resize(expanded.size() + pack->size() - 1);
            for(auto p : *pack) {
                expanded[cur_idx] = p;
                ++cur_idx;
            }
        } else {
            expanded[cur_idx] = array->getParams()[i];
            ++cur_idx;
        }
    }
    return expanded;
}


int main() {
    // const char* name =
    // "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_22CUDAFunctorOnOther_addIbEENS_6detail5ArrayIPcLi2EEEEEviT0_T1_";
    // const char* name =
    // "_ZN2at6native18elementwise_kernelILi128ELi4EZNS0_15gpu_kernel_implINS0_22CUDAFunctorOnOther_addIbEEEEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1_";
    const char *name =
        "_ZN2at6native51_GLOBAL__N__49690f20_18_fused_adam_impl_cu_"
        "ffd69f6925multi_tensor_apply_kernelINS1_"
        "32FusedOptimizerTensorListMetadataILi4EEENS1_"
        "20FusedAdamMathFunctorIdLi4EEEJdddddbbPfS7_EEEvT_T0_DpT1_";
    Demangler Parser(name, name + std::strlen(name));
    auto *AST =
        dynamic_cast<itanium_demangle::FunctionEncoding *>(Parser.parse());
    std::vector<itanium_demangle::Node*> expanded_parameters = expand_mangled_par(AST);

    size_t N_params = expanded_parameters.size();
    std::vector<int> is_ptr(N_params);



    for (unsigned i = 0; i < N_params; ++i) {
        auto par = expanded_parameters[i];

        switch (par->getKind()) {
        case itanium_demangle::Node::KNameWithTemplateArgs:
            deduce_nested_ptr(
                dynamic_cast<itanium_demangle::NameWithTemplateArgs *>(par),
                is_ptr[i]);
            break;
        case itanium_demangle::Node::KPointerType:
            is_ptr[i] = true;
            break;
        default:
            is_ptr[i] = false;
        }
    }

    std::cout << "Pointers: ";
    for (auto p : is_ptr) {
        if (p)
            std::cout << "yes ";
        else
            std::cout << "no ";
    }
    std::cout << std::endl;
}